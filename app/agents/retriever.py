from typing import Dict, Any
from app.agents.base import BaseAgent
from app.services.vector_store_service import VectorStoreService

class RetrieverAgent(BaseAgent):
    """Agent responsible for semantic retrieval of relevant documents"""
    
    def __init__(self):
        super().__init__("retriever_agent")
        self.vector_service = VectorStoreService()
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant documents based on query"""
        query = state.get("query", "")
        
        if not query:
            return {"retrieved_docs": [], "retrieval_error": "No query provided"}
        
        try:
            # Perform semantic search
            documents = self.vector_service.similarity_search(query)
            
            # Extract content and metadata
            retrieved_docs = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": getattr(doc, 'score', 0.0)
                }
                for doc in documents
            ]
            
            # Create context string for LLM
            context = "\n\n".join([
                f"Document {i+1}: {doc['content']}" 
                for i, doc in enumerate(retrieved_docs)
            ])
            
            return {
                "retrieved_docs": retrieved_docs,
                "context": context,
                "num_retrieved": len(retrieved_docs)
            }
            
        except Exception as e:
            return {
                "retrieved_docs": [],
                "retrieval_error": str(e),
                "context": ""
            }