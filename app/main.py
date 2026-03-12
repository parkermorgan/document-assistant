from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from openai import OpenAI
import chromadb
import anthropic

load_dotenv()

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
chroma_client = chromadb.PersistentClient(path="data/chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

openai_api_key = os.getenv("OPENAI_API_KEY")

print("ANTHROPIC KEY LOADED:", os.getenv("ANTHROPIC_API_KEY"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

class QueryRequest(BaseModel):
    question: str


# GET requests
@app.get("/")
def health_check():
    return {"status": "ok"}

@app.get("/hello")
def say_hello():
    return {"message": "Hello from document assistant!"}

@app.get("/hello/{name}")
def say_hello_to(name: str):
    return {"message": f"Hello {name}!"}

@app.get("/test-embedding")
def test_embedding():
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="Python developer with AI expereience"
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

# POST requests
@app.post("/query")
def query(request: QueryRequest):

    # Embed the question
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=request.question
    )
    question_embedding = response.data[0].embedding

    # Search for most relevant chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    # Extract matched chunks
    matched_chunks = results["documents"][0]

    # Build context from chunks
    context = "\n\n".join(matched_chunks)

    # Ask Claude
    message = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Answer the question based on on the context provided below.""
Context:
{context}

Question: {request.question}"""
            }
        ]
    )

    return {
        "question": request.question,
        "answer": message.content[0].text,
        "sources": matched_chunks
    }

# Upload a file
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted"
        )

    contents = await file.read()

    # Open the PDF from bytes
    pdf = fitz.open(stream=contents, filetype="pdf")

    # Extract text
    text = ""
    for page in pdf:
        text += page.get_text()

    # Chunk the text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)

    # Embed each chunck
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )
    embeddings = [item.embedding for item in response.data]

    # Store in chromadb
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{file.filename}-chunk-{i}" for i in range(len(chunks))]
    )

    return {
        "filename": file.filename,
        "pages": len(pdf),
        "chunks_stored": len(chunks)
    }