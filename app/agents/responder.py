from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from app.agents.base import BaseAgent
from app.config import get_settings


class ResponderAgent(BaseAgent):
    """Agent responsible for generating responses based on retrieved context"""

    def __init__(self):
        super().__init__("responder_agent")
        self.settings = get_settings()
        self.llm = ChatOpenAI(
            model=self.settings.chat_model,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            openai_api_key=self.settings.openai_api_key,
        )

    def _build_system_prompt(self) -> str:
        """Build the system prompt for response generation"""
        return """You are a helpful product assistant for an e-commerce platform. Your role is to:

1. Answer questions about products based ONLY on the provided context
2. If the context doesn't contain relevant information, politely say you don't have that information
3. Be conversational and helpful, but stay grounded in facts
4. For greetings or casual conversation, respond naturally but briefly
5. Always prioritize accuracy over helpfulness - don't make up product details

Guidelines:
- Use the product information from the context to answer queries
- Be specific about sizes, colors, prices when available
- If asked about availability, refer to what's mentioned in the context
- Don't hallucinate or invent product details not in the context
"""

    def _build_user_prompt(self, query: str, context: str) -> str:
        """Build the user prompt with query and context"""
        if context.strip():
            return f"""Context (Retrieved product information):
{context}

User Query: {query}

Please provide a helpful response based on the context above."""
        else:
            return f"""User Query: {query}

Note: No specific product context was retrieved. Please respond appropriately to the user's query."""

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response based on retrieved context and user query"""
        query = state.get("query", "")
        context = state.get("context", "")

        # Build messages for the LLM
        messages = [
            SystemMessage(content=self._build_system_prompt()),
            HumanMessage(content=self._build_user_prompt(query, context)),
        ]

        try:
            # Generate response
            response = self.llm.invoke(messages)

            # Calculate a simple confidence score based on context availability
            confidence_score = 0.9 if context.strip() else 0.3
            if state.get("retrieval_error"):
                confidence_score = 0.1

            return {
                "answer": response.content,
                "confidence_score": confidence_score,
                "processing_successful": True,
            }

        except Exception as e:
            return {
                "answer": "I apologize, but I'm having technical difficulties processing your request. Please try again later.",
                "confidence_score": 0.0,
                "processing_error": str(e),
                "processing_successful": False,
            }
