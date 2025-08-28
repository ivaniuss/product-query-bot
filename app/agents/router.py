from typing import Literal
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from app.config import get_settings

class RouterAgent:
    """Intelligent intent router using LLM with optimization"""
    
    def __init__(self):
        self.settings = get_settings()
        self.llm = ChatOpenAI(
            model=self.settings.chat_model,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            openai_api_key=self.settings.openai_api_key
        )
        # Cache to avoid repeated API calls for same queries
        self._classification_cache = {}
        # Statistics for monitoring
        self._cache_hits = 0
        self._llm_calls = 0
    
    def classify_intent(self, query: str) -> Literal["product_query", "general_conversation"]:
        """
        Classify user intent with multi-layer approach:
        1. Cache lookup (fastest)
        2. Heuristic rules (fast)
        3. LLM classification (accurate but slower)
        """
        
        # Layer 1: Check cache
        cache_key = query.lower().strip()
        if cache_key in self._classification_cache:
            self._cache_hits += 1
            return self._classification_cache[cache_key]
        
        # Layer 2: Fast heuristics for obvious cases
        heuristic_result = self._apply_heuristics(query)
        if heuristic_result:
            self._classification_cache[cache_key] = heuristic_result
            return heuristic_result
        
        # Layer 3: LLM classification for ambiguous cases
        llm_result = self._llm_classify(query)
        self._classification_cache[cache_key] = llm_result
        return llm_result
    
    def _apply_heuristics(self, query: str) -> Literal["product_query", "general_conversation", None]:
        """Apply fast heuristic rules for obvious cases"""
        
        query_clean = query.lower().strip().rstrip('!?.,')
        
        # Obvious greetings and social interactions
        definite_greetings = {
            "hi", "hello", "hey", "hola", "good morning", "good afternoon", 
            "good evening", "goodbye", "bye", "thanks", "thank you",
            "how are you", "what's up", "nice to meet you"
        }
        
        if query_clean in definite_greetings:
            return "general_conversation"
        
        # Check if starts with greeting
        greeting_starters = ["hi ", "hello ", "hey ", "good morning", "good afternoon"]
        if any(query_clean.startswith(starter) for starter in greeting_starters):
            return "general_conversation"
        
        # Strong product/commerce indicators
        strong_product_signals = [
            "$", "price", "cost", "buy", "purchase", "order", "cart",
            "available", "stock", "inventory", "size", "color", 
            "brand", "model", "specification", "spec", "delivery",
            "shipping", "discount", "sale", "offer"
        ]
        
        if any(signal in query.lower() for signal in strong_product_signals):
            return "product_query"
        
        # Questions about products (common patterns)
        product_question_patterns = [
            "do you have", "can i get", "show me", "find me",
            "what's available", "any", "which", "recommend"
        ]
        
        query_lower = query.lower()
        if any(pattern in query_lower for pattern in product_question_patterns):
            # Could be product or general - let LLM decide
            return None
        
        # If nothing matches heuristics, use LLM
        return None
    
    def _llm_classify(self, query: str) -> Literal["product_query", "general_conversation"]:
        """Use LLM for nuanced classification"""
        
        system_prompt = """You are a precise intent classifier for an e-commerce chatbot.

Classify user queries into exactly one category:

🛍️ PRODUCT: Questions about products, shopping, items, prices, availability, specifications, purchases, or any commerce-related intent.

💬 CHAT: Greetings, casual conversation, questions about the service itself, gratitude, or social interactions.

Key principles:
- "Do you have X?" = PRODUCT (even if X is general)
- "What can you help with?" = CHAT (asking about service capabilities)
- "Hello, do you have shoes?" = PRODUCT (intent is product despite greeting)
- When in doubt, choose PRODUCT (better to show products than miss a potential customer)

Examples:
"Do you have Nike shoes?" → PRODUCT
"What sizes do you carry?" → PRODUCT  
"Hello!" → CHAT
"How does this work?" → CHAT
"Thanks for helping!" → CHAT
"Any running shoes available?" → PRODUCT

Respond with exactly one word: PRODUCT or CHAT"""

        try:
            self._llm_calls += 1
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Query: {query}")
            ]
            
            response = self.llm.invoke(messages)
            classification = response.content.strip().upper()
            
            # Parse response
            if "PRODUCT" in classification:
                return "product_query"
            elif "CHAT" in classification:
                return "general_conversation"
            else:
                # Default to product query for business reasons
                # (better to show products than miss a potential sale)
                print(f"Ambiguous LLM response '{classification}' for query '{query}', defaulting to product_query")
                return "product_query"
                
        except Exception as e:
            print(f"LLM classification failed for query '{query}': {e}")
            # Safe default: assume product query to be helpful
            return "product_query"
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        total_classifications = self._cache_hits + self._llm_calls
        cache_hit_rate = (self._cache_hits / total_classifications * 100) if total_classifications > 0 else 0
        
        return {
            "total_classifications": total_classifications,
            "cache_hits": self._cache_hits,
            "llm_calls": self._llm_calls,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "cached_queries": len(self._classification_cache)
        }
    
    def clear_cache(self):
        """Clear classification cache"""
        self._classification_cache.clear()
        self._cache_hits = 0
        self._llm_calls = 0