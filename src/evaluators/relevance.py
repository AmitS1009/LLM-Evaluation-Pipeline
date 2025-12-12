import os
from openai import OpenAI
from .base import BaseEvaluator
from ..models import Conversation, Context

class RelevanceEvaluator(BaseEvaluator):
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def evaluate(self, conversation: Conversation, context: Context) -> dict:
        messages = conversation.messages
        if len(messages) < 2:
            return {"relevance_score": 0.0, "relevance_reasoning": "Insufficient messages"}

        # Use new model fields: .message instead of .content
        user_query = messages[-2].message 
        assistant_response = messages[-1].message

        prompt = f"""
        You are an expert evaluator. 
        User Query: "{user_query}"
        AI Response: "{assistant_response}"

        Task: Evaluate the AI response for Relevance and Completeness based on the User Query.
        1. Relevance: Does it directly address the user's intent?
        2. Completeness: Does it answer all parts of the query?

        Output STRICT JSON format:
        {{
            "score": <float between 0.0 and 1.0>,
            "reasoning": "<brief explanation>"
        }}
        """

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "You are a helpful evaluator."},
                          {"role": "user", "content": prompt}],
                temperature=0.0
            )
            content = completion.choices[0].message.content
            import json
            import re
            
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                return {
                    "relevance_score": data.get("score", 0.0),
                    "relevance_reasoning": data.get("reasoning", "No reasoning provided")
                }
            else:
                return {"relevance_score": 0.0, "relevance_reasoning": "Failed to parse LLM output"}

        except Exception as e:
            return {"relevance_score": 0.0, "relevance_reasoning": f"Error: {str(e)}"}
