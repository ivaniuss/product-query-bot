import os
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from app.config import get_settings

class VectorStoreService:
    def __init__(self):
        self.settings = get_settings()
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.embedding_model,
            openai_api_key=self.settings.openai_api_key
        )
        self.vectorstore = None
        self._load_or_create_vectorstore()
    
    def _load_or_create_vectorstore(self):
        """Load existing vectorstore or create new one"""
        vector_path = self.settings.vector_store_path
        
        if os.path.exists(f"{vector_path}.faiss"):
            try:
                self.vectorstore = FAISS.load_local(
                    vector_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("Loaded existing vector store")
            except Exception as e:
                print(f"Failed to load vector store: {e}")
                self._create_default_vectorstore()
        else:
            self._create_default_vectorstore()
    
    def _create_default_vectorstore(self):
        """Create vectorstore with sample product data"""
        sample_products = [
            "Nike Air Max 270 sneakers, size 42, black/white colorway, $120, breathable mesh upper",
            "Adidas Ultraboost 22 running shoes, size 41, grey/blue, $180, responsive cushioning",
            "Converse Chuck Taylor All Star, size 39, red canvas, $65, classic high-top design",
            "Vans Old Skool sneakers, size 43, black/white, $60, durable suede and canvas upper",
            "New Balance 990v5, size 40, grey, $175, premium made in USA construction",
            "Puma RS-X sneakers, size 42, white/black/red, $110, retro-inspired chunky sole",
            "Jordan 1 Retro High, size 44, bred colorway, $170, premium leather construction",
            "Reebok Classic Leather, size 38, white, $75, soft garment leather upper",
            "ASICS Gel-Kayano 29, size 41, blue/silver, $160, stability running shoe",
            "Skechers Go Walk 6, size 39, black, $80, ultra-lightweight walking shoe"
        ]
        
        documents = [Document(page_content=text, metadata={"source": f"product_{i}"}) 
                    for i, text in enumerate(sample_products)]
        
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # Save the vectorstore
        os.makedirs(os.path.dirname(self.settings.vector_store_path), exist_ok=True)
        self.vectorstore.save_local(self.settings.vector_store_path)
        print("Created and saved new vector store with sample products")
    
    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Perform similarity search"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        k = k or self.settings.top_k
        return self.vectorstore.similarity_search(query, k=k)
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to the vector store"""
        if not self.vectorstore:
            self._create_default_vectorstore()
        
        self.vectorstore.add_documents(documents)
        self.vectorstore.save_local(self.settings.vector_store_path)