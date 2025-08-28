from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from app.agents.retriever import RetrieverAgent  
from app.agents.responder import ResponderAgent
from app.agents.router import RouterAgent

class AgentState(dict):
    """State class for the agent workflow"""
    user_id: str
    query: str
    intent: str = ""  # Add intent tracking
    retrieved_docs: list = []
    context: str = ""
    answer: str = ""
    confidence_score: float = 0.0
    processing_successful: bool = False

class MultiAgentWorkflow:
    """Multi-agent workflow with intelligent routing"""
    
    def __init__(self):
        self.retriever_agent = RetrieverAgent()
        self.responder_agent = ResponderAgent()
        self.intent_router = RouterAgent()
        self.checkpointer = InMemorySaver()
        self.app = self._build_workflow()
    
    def _classify_and_route(self, state: AgentState) -> Literal["retriever", "responder"]:
        """Classify intent and route to appropriate agent"""
        query = state.get("query", "")
        
        if not query.strip():
            # Empty query goes directly to responder
            state["intent"] = "empty_query"
            return "responder"
        
        # Use smart router to classify intent
        intent = self.intent_router.classify_intent(query)
        state["intent"] = intent  # Store intent in state for debugging/analytics
        
        # Route based on classification
        if intent == "product_query":
            return "retriever"
        else:  # general_conversation
            return "responder"
    
    def _retriever_node(self, state: AgentState) -> AgentState:
        """Execute retriever agent with intent context"""
        result = self.retriever_agent.execute(state)
        state.update(result)
        return state
    
    def _responder_node(self, state: AgentState) -> AgentState:
        """Execute responder agent with intent awareness"""
        # Add intent information to help responder
        if "intent" not in state:
            state["intent"] = "unknown"
            
        result = self.responder_agent.execute(state)
        state.update(result)
        return state
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with smart routing"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("responder", self._responder_node)
        
        # Smart conditional routing from START
        workflow.add_conditional_edges(
            START,
            self._classify_and_route,
            {
                "retriever": "retriever",
                "responder": "responder"
            }
        )
        
        # Retriever always flows to responder
        workflow.add_edge("retriever", "responder")
        
        # Responder ends the workflow
        workflow.add_edge("responder", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def process_query(self, user_id: str, query: str) -> Dict[str, Any]:
        """Process a query through the multi-agent workflow"""
        config = {"configurable": {"thread_id": user_id}}
        
        initial_state = AgentState({
            "user_id": user_id,
            "query": query
        })
        
        try:
            result = self.app.invoke(initial_state, config)
            
            # Include routing information in response for debugging
            return {
                "answer": result.get("answer", ""),
                "confidence_score": result.get("confidence_score", 0.0),
                "retrieved_docs": result.get("retrieved_docs", []),
                "processing_successful": result.get("processing_successful", False),
                "intent": result.get("intent", "unknown"),  # For analytics
                "routing_stats": self.intent_router.get_stats()  # Performance metrics
            }
            
        except Exception as e:
            return {
                "answer": "I apologize, but I encountered an error processing your request.",
                "confidence_score": 0.0,
                "retrieved_docs": [],
                "processing_successful": False,
                "error": str(e),
                "intent": "error"
            }
    
    def get_routing_performance(self) -> Dict[str, Any]:
        """Get routing performance statistics"""
        return self.intent_router.get_stats()
    
    def clear_routing_cache(self):
        """Clear the routing cache"""
        self.intent_router.clear_cache()
        
workflow = MultiAgentWorkflow()