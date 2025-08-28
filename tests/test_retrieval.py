import pytest
from unittest.mock import Mock, patch
from app.agents.retriever import RetrieverAgent
from langchain.schema import Document

class TestRetrieverAgent:
    
    @pytest.fixture
    def retriever_agent(self):
        with patch('app.agents.retriever.VectorStoreService'):
            return RetrieverAgent()
    
    def test_execute_with_valid_query(self, retriever_agent):
        # Mock the vector service
        mock_docs = [
            Document(page_content="Nike shoes size 42", metadata={"source": "product_1"})
        ]
        retriever_agent.vector_service.similarity_search = Mock(return_value=mock_docs)
        
        state = {"query": "Nike shoes"}
        result = retriever_agent.execute(state)
        
        assert "retrieved_docs" in result
        assert "context" in result
        assert len(result["retrieved_docs"]) == 1
        assert "Nike shoes size 42" in result["context"]
    
    def test_execute_with_empty_query(self, retriever_agent):
        state = {"query": ""}
        result = retriever_agent.execute(state)
        
        assert "retrieval_error" in result
        assert result["retrieved_docs"] == []