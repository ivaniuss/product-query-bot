from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse
from app.agents.workflow import workflow

router = APIRouter(prefix="/api", tags=["queries"])

@router.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest) -> QueryResponse:
    """
    Handle user queries about products using multi-agent RAG pipeline
    """
    try:
        # Process query through multi-agent workflow
        result = workflow.process_query(request.user_id, request.query)
        
        if not result.get("processing_successful", False):
            raise HTTPException(
                status_code=500, 
                detail=f"Query processing failed: {result.get('error', 'Unknown error')}"
            )
        
        return QueryResponse(
            answer=result["answer"],
            retrieved_docs=[doc.get("content", "") for doc in result.get("retrieved_docs", [])],
            confidence_score=result.get("confidence_score")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "product-query-bot"}