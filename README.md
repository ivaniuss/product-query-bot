# Product Query Bot - Multi-Agent RAG System

A microservice that handles product queries using a multi-agent Retrieval-Augmented Generation (RAG) pipeline built with LangGraph and FastAPI.

## 🏗️ Architecture

The system implements a multi-agent architecture with clean separation of concerns:

- **Retriever Agent**: Handles semantic document retrieval using vector embeddings
- **Responder Agent**: Generates contextual responses using retrieved documents
- **Router Agent**: Routes queries to appropriate agents based on content (product queries vs. greetings)
- **Vector Store**: FAISS-based in-memory vector database for document embeddings

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- Docker (optional)

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/ivaniuss/product-query-bot.git
cd product-query-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```
#### 🔧 Configuration

Environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `EMBEDDING_MODEL` | Embedding model name | `text-embedding-3-small` |
| `CHAT_MODEL` | Chat completion model | `gpt-3.5-turbo` |
| `TOP_K` | Number of documents to retrieve | `3` |
| `TEMPERATURE` | LLM temperature | `0.1` |
| `MAX_TOKENS` | Maximum response tokens | `500` |
| `VECTOR_STORE_PATH` | Vector store file path | `./data/vector_store` |

### 2. Running the Application

#### Option A: Direct Python

```bash
# Run the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

#### Option B: Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or with plain Docker
docker build -t product-query-bot .
docker run -p 8000:8000 --env-file .env product-query-bot
```

## 🧪 Testing the System

### Manual Testing

```bash
# Make the script executable
chmod +x example_requests.sh

# Run example requests
./example_requests.sh
```

### Example Requests

**Health Check:**
```bash
curl -X GET "http://localhost:8000/api/health"
```

**Product Query:**
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123", 
    "query": "Do you have Nike Air Max shoes in size 42?"
  }'
```

**Greeting:**
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_456", 
    "query": "Hello!"
  }'
```

### Automated Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_retrieval.py -v
```

## 📊 API Documentation

### POST /api/query

**Request Body:**
```json
{
  "user_id": "string",
  "query": "string"
}
```

**Response:**
```json
{
  "answer": "string",
  "retrieved_docs": ["string"],
  "confidence_score": 0.85
}
```

**Status Codes:**
- `200`: Successful response
- `422`: Validation error
- `500`: Internal server error

## ⏱️ Time Spent 
- ~ 8 hours