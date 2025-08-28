from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.query import router as query_router
from app.config import get_settings

def create_app() -> FastAPI:
    get_settings()
    
    app = FastAPI(
        title="Product Query Bot",
        description="Multi-agent RAG system for product queries",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(query_router)
    
    @app.get("/")
    async def root():
        return {"message": "Product Query Bot API", "docs": "/docs"}
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)