# Document Assistant

An agent-based document Q&A system that lets you upload PDF documents and ask natural language questions about them. Built with a production-realistic RAG architecture using summary embeddings, semantic search, and Claude-powered tool use.

# Demonstration
Youtube Demo: https://youtu.be/0z8sKceHkBQ

## Architecture

Rather than embedding raw document chunks (a common but brittle approach), this project embeds **AI-generated summaries** of each document. When a question is asked, the agent uses semantic search to find the most relevant documents, then reads the full document content to form an accurate answer.

This approach keeps embeddings accurate over time — summaries don't go stale the way chunked content can — and separates the concerns of *finding* the right document from *reading* it.

### Agent Tools

The Claude-powered agent has access to three tools:

- **`semantic_search`** — embeds the user's question and finds the most semantically relevant documents in ChromaDB
- **`read_document`** — reads the full text of a document from disk given its file path
- **`keyword_search`** — searches document summaries for exact keyword or phrase matches

The agent decides which tools to call, in what order, based on the question. It always searches before answering and never responds from memory alone.

### Upload Flow

1. User uploads a PDF
2. Text is extracted using PyMuPDF
3. File is saved to local storage
4. Claude generates a concise summary of the document
5. Summary is embedded using OpenAI `text-embedding-3-small`
6. Embedding + metadata (filename, file path, upload date) stored in ChromaDB

### Query Flow

1. User submits a question
2. Question is embedded and compared against document summary vectors in ChromaDB
3. Agent reads full content of relevant documents using the `read_document` tool
4. Claude forms a natural language answer grounded in the document content

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| LLM & Agent | Anthropic Claude (claude-sonnet-4) |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Database | ChromaDB |
| PDF Parsing | PyMuPDF (fitz) |
| Frontend | Streamlit |
| Containerization | Docker + Docker Compose |

## Project Structure

```
document-assistant/
├── app/
│   ├── __init__.py
│   └── main.py          # FastAPI app, agent loop, tool functions
├── frontend/
│   └── streamlit_app.py # Streamlit UI
├── data/
│   ├── documents/       # uploaded PDFs (gitignored)
│   └── chroma_db/       # vector store (gitignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.11
- Docker and Docker Compose
- OpenAI API key
- Anthropic API key

### Running with Docker (recommended)

1. Clone the repository:
```bash
git clone https://github.com/parkermorgan/document-assistant.git
cd document-assistant
```

2. Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

3. Add your API keys to `.env`:
```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

4. Start the app:
```bash
docker compose up --build
```

5. Open the Streamlit UI at `http://localhost:8501`

### Running Locally

1. Create and activate a virtual environment:
```bash
python3.11 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create your `.env` file with API keys (see above)

4. Start the API:
```bash
uvicorn app.main:app --reload --reload-dir app
```

5. In a separate terminal, start the frontend:
```bash
streamlit run frontend/streamlit_app.py
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/upload` | Upload and index a PDF document |
| POST | `/query` | Ask a question about uploaded documents |

### Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What programming languages does this person know?"}'
```

## Design Decisions

**Summary embeddings over chunk embeddings** — Embedding document summaries rather than raw chunks keeps the vector store accurate over time. Chunk-based approaches require constant re-embedding as documents change; summaries remain valid much longer and only need periodic refreshing.

**Agent-based retrieval** — Using Claude as an agent with discrete tools (search, read, keyword match) produces more reliable answers than a single retrieval step. The agent can chain tool calls, cross-reference documents, and fall back to keyword search when semantic search is insufficient.

**FastAPI + Streamlit separation** — Keeping the API and frontend as separate services makes it easy to swap the frontend later (a React app, a CLI, another API consumer) without touching the backend logic.
