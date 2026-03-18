from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from openai import OpenAI
import chromadb
import anthropic
import shutil
from datetime import datetime

load_dotenv()

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
chroma_client = chromadb.PersistentClient(path="data/chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

class QueryRequest(BaseModel):
    question: str


tools = [
    {
        "name": "read_document",
        "description": "Reads the full text content of a document given its file path. Use this when you need to read the actual content of a specific document to answer a question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The file path to the document to read"
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "semantic_search",
        "description": "Searches document summaries using semantic similarity. Use this to find which documents are most relevant to a question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant documents"
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return. Default 3."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "keyword_search",
        "description": "Searches for exact keyword matches across all stored document summaries. Use this when looking for specific terms, names, or phrases.",
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "The keyword or phrase to search for"
                }
            },
            "required": ["keyword"]
        }
    }
]

@app.delete("/clear")
def clear_collection():
    chroma_client.delete_collection("documents")
    globals()["collection"] = chroma_client.get_or_create_collection(name="documents")
    return {"status": "collection cleared"}

# GET requests
@app.get("/")
def health_check():
    return {"status": "ok"}

@app.get("/test-embedding")
def test_embedding():
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="Python developer with AI experience"
    )
    vector = response.data[0].embedding
    return {
        "text": "Python developer with AI experience",
        "vector_length": len(vector),
        "first_5_numbers": vector[:5]
    }

@app.get("/test-similarity")
def test_similarity():
    texts = [
        "Python developer with AI experience",
        "software engineer who codes in Python",
        "I enjoy cooking pasta"
    ]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    vectors = [item.embedding for item in response.data]

    def similarity(a, b):
        return sum(x * y for x, y in zip(a, b))

    return {
        "python_dev_vs_software_engineer": similarity(vectors[0], vectors[1]),
        "python_dev_vs_cooking_pasta": similarity(vectors[0], vectors[2])
    }


# Tool functions
def run_read_document(file_path: str) -> str:
    try:
        pdf = fitz.open(file_path)
        text = ""
        for page in pdf:
            text += page.get_text()
        return text
    except Exception as e:
        return f"Error reading document: {str(e)}"


def run_semantic_search(query: str, n_results: int = 3) -> str:
    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = embedding_response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    if not results["documents"][0]:
        return "No relevant documents found."

    output = []
    for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        output.append(f"Document {i+1}: {metadata['filename']}\nFile path: {metadata['file_path']}\nSummary: {doc}")

    return "\n\n".join(output)


def run_keyword_search(keyword: str) -> str:
    all_results = collection.get()

    if not all_results["documents"]:
        return "No documents in the library."

    matches = []
    for doc, metadata in zip(all_results["documents"], all_results["metadatas"]):
        if keyword.lower() in doc.lower():
            matches.append(f"Document: {metadata['filename']}\nFile path: {metadata['file_path']}\nSummary: {doc}")

    if not matches:
        return f"No documents found containing '{keyword}'"

    return "\n\n".join(matches)


def run_agent(question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": question
        }
    ]

    system_prompt = """You are a helpful document assistant. You have access to a library of documents.
When answering questions:
1. First use semantic_search to find relevant documents
2. Use read_document to read the full content of relevant documents
3. Use keyword_search if you need to find specific terms or names
4. Answer based on what you find in the documents

Always search before answering - never answer from memory alone."""

    while True:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            tools=tools,
            messages=messages
        )

        messages.append({
            "role": "assistant",
            "content": response.content
        })

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text

        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input

                    if tool_name == "read_document":
                        result = run_read_document(**tool_input)
                    elif tool_name == "semantic_search":
                        result = run_semantic_search(**tool_input)
                    elif tool_name == "keyword_search":
                        result = run_keyword_search(**tool_input)
                    else:
                        result = f"Unknown tool: {tool_name}"

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({
                "role": "user",
                "content": tool_results
            })


# POST requests
@app.post("/query")
def query(request: QueryRequest):
    answer = run_agent(request.question)
    return {"question": request.question, "answer": answer}


# Upload a file
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted"
        )

    contents = await file.read()

    file_path = f"data/documents/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(contents)

    pdf = fitz.open(stream=contents, filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()

    summary_response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Please provide a concise summary of this document.
Include the main topics, key information, and what questions this document would be useful for answering.

Document:
{text}"""
            }
        ]
    )
    summary = summary_response.content[0].text

    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=summary
    )
    embedding = embedding_response.data[0].embedding

    collection.add(
        documents=[summary],
        embeddings=[embedding],
        ids=[file.filename],
        metadatas=[{
            "filename": file.filename,
            "file_path": file_path,
            "upload_date": datetime.now().isoformat()
        }]
    )

    return {
        "filename": file.filename,
        "file_path": file_path,
        "summary": summary
    }