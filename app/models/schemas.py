from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    query: str = Field(..., min_length=1, description="User's query about products")

class QueryResponse(BaseModel):
    answer: str
    retrieved_docs: Optional[List[str]] = None
    confidence_score: Optional[float] = None

class Document(BaseModel):
    content: str
    metadata: dict = {}