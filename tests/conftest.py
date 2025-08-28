import os
import pytest
from fastapi.testclient import TestClient
from app.main import create_app

@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables"""
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-small")
    os.environ.setdefault("CHAT_MODEL", "gpt-3.5-turbo")

@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app"""
    app = create_app()
    with TestClient(app) as client:
        yield client
