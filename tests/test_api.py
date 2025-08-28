import pytest
from httpx import AsyncClient
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

class TestAPI:
    
    def test_health_endpoint_sync(self, test_client):
        """Test health endpoint with synchronous client"""
        response = test_client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_health_endpoint_async(self):
        """Test health endpoint with async client"""
        # Import after environment is set
        from app.main import create_app
        from httpx import ASGITransport
        
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "service" in data
    
    def test_query_endpoint_validation(self, test_client):
        """Test query endpoint input validation"""
        # Test missing fields
        response = test_client.post("/api/query", json={})
        assert response.status_code == 422
        
        # Test empty query
        response = test_client.post("/api/query", json={
            "user_id": "test_user",
            "query": ""
        })
        assert response.status_code == 422
    
    @patch('app.agents.workflow.MultiAgentWorkflow.process_query')
    def test_query_endpoint_success(self, mock_process, test_client):
        """Test successful query processing"""
        # Mock the workflow response
        mock_process.return_value = {
            "answer": "We have Nike Air Max shoes available in size 42.",
            "confidence_score": 0.85,
            "retrieved_docs": [{"content": "Nike Air Max size 42 available"}],
            "processing_successful": True
        }
        
        payload = {
            "user_id": "test_user",
            "query": "Do you have Nike shoes in size 42?"
        }
        
        response = test_client.post("/api/query", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert "confidence_score" in data
        assert data["confidence_score"] == 0.85
    
    @patch('app.agents.workflow.MultiAgentWorkflow.process_query')
    def test_query_endpoint_processing_failure(self, mock_process, test_client):
        """Test query processing failure handling"""
        # Mock processing failure
        mock_process.return_value = {
            "answer": "Error occurred",
            "confidence_score": 0.0,
            "retrieved_docs": [],
            "processing_successful": False,
            "error": "Mock error"
        }
        
        payload = {
            "user_id": "test_user",
            "query": "test query"
        }
        
        response = test_client.post("/api/query", json=payload)
        assert response.status_code == 500
        assert "Query processing failed" in response.json()["detail"]
