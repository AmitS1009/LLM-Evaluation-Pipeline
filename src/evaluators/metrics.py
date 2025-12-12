import tiktoken
from .base import BaseEvaluator
from ..models import Conversation, Context

class MetricsEvaluator(BaseEvaluator):
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.pricing = {
            "gpt-4": (0.03, 0.06),
            "gpt-3.5-turbo": (0.0005, 0.0015),
            "gpt-4o": (0.005, 0.015),
            "gpt-4o-mini": (0.00015, 0.0006)
        }

    def _count_tokens(self, text: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def evaluate(self, conversation: Conversation, context: Context) -> dict:
        messages = conversation.messages # Uses the property accessor
        if not messages:
            return {"latency_ms": None, "estimated_cost_usd": 0.0}

        # 1. Latency Calculation
        # Identify the LAST turn where AI responded.
        latency = None
        if len(messages) >= 2:
            last_msg = messages[-1]
            prev_msg = messages[-2]
            
            # Roles in new JSON: "User", "AI/Chatbot"
            # Normalize for checking
            last_role = last_msg.role.lower()
            prev_role = prev_msg.role.lower()

            if "ai" in last_role and "user" in prev_role:
                if last_msg.created_at and prev_msg.created_at:
                    diff = last_msg.created_at - prev_msg.created_at
                    latency = diff.total_seconds() * 1000

        # 2. Cost Calculation
        input_tokens = 0
        output_tokens = 0
        
        for msg in messages:
            count = self._count_tokens(msg.message) # Field is 'message' now
            if "ai" in msg.role.lower():
                output_tokens += count
            else:
                input_tokens += count
        
        for chunk in context.context_chunks:
            input_tokens += self._count_tokens(chunk)

        input_price, output_price = self.pricing.get(conversation.model, self.pricing["gpt-4"])
        cost = (input_tokens / 1000 * input_price) + (output_tokens / 1000 * output_price)

        return {
            "latency_ms": latency,
            "estimated_cost_usd": cost,
            "token_usage": {"input": input_tokens, "output": output_tokens}
        }
