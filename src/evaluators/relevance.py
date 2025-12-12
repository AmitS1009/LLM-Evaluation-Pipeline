import os
import json
import re
from openai import OpenAI
import google.generativeai as genai
from .base import BaseEvaluator
from ..models import Conversation, Context

class RelevanceEvaluator(BaseEvaluator):
    def __init__(self, model_openai: str = "gpt-3.5-turbo", model_gemini: str = "gemini-pro"):
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        
        self.provider = "openai" if self.openai_key else "gemini" if self.gemini_key else None
        
        if self.provider == "openai":
            self.client = OpenAI(api_key=self.openai_key)
            self.model = model_openai
        elif self.provider == "gemini":
            genai.configure(api_key=self.gemini_key)
            self.model = model_gemini
        else:
            print("Warning: No API Keys found for RelevanceEvaluator (OpenAI or Gemini)")

    def evaluate(self, conversation: Conversation, context: Context) -> dict:
        messages = conversation.messages
        if len(messages) < 2:
            return {"relevance_score": 0.0, "relevance_reasoning": "Insufficient messages"}

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
            if self.provider == "openai":
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": "You are a helpful evaluator."},
                              {"role": "user", "content": prompt}],
                    temperature=0.0
                )
                content = completion.choices[0].message.content
            elif self.provider == "gemini":
                model_instance = genai.GenerativeModel(self.model)
                response = model_instance.generate_content(prompt)
                content = response.text
            else:
                 return {"relevance_score": 0.0, "relevance_reasoning": "No valid API provider configured"}

            # Parse JSON
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                json_str = match.group(0)
                # Cleanup common JSON issues if any
                json_str = re.sub(r",\s*(?=[\]}])", "", json_str) 
                data = json.loads(json_str)
                return {
                    "relevance_score": data.get("score", 0.0),
                    "relevance_reasoning": data.get("reasoning", "No reasoning provided")
                }
            else:
                return {"relevance_score": 0.0, "relevance_reasoning": "Failed to parse LLM output"}

        except Exception as e:
            return {"relevance_score": 0.0, "relevance_reasoning": f"Error: {str(e)}"}
