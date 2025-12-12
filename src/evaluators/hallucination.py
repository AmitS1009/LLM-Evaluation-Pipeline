import os
import json
import re
from openai import OpenAI
import google.generativeai as genai
from .base import BaseEvaluator
from ..models import Conversation, Context

class HallucinationEvaluator(BaseEvaluator):
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
             print("Warning: No API Keys found for HallucinationEvaluator (OpenAI or Gemini)")

    def evaluate(self, conversation: Conversation, context: Context) -> dict:
        messages = conversation.messages
        if len(messages) < 1:
            return {"hallucination_score": 0.0, "hallucination_reasoning": "No response to evaluate"}

        assistant_response = messages[-1].message
        context_text = "\n\n".join(context.context_chunks)

        prompt = f"""
        You are a fact-checking assistant.
        Context:
        {context_text}

        AI Response:
        {assistant_response}

        Task: Determine if the AI Response contains any hallucinations or factual inaccuracies based STRICTLY on the Context provided. 
        If the response mentions something not in the context but is general knowledge, mark it as 'Faithful' IF it doesn't contradict the context.
        Output STRICT JSON format:
        {{
            "score": <float 0.0 (hallucinated) to 1.0 (faithful)>,
            "reasoning": "<brief explanation>"
        }}
        """

        try:
            if self.provider == "openai":
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": "You are a strict fact-checker."},
                              {"role": "user", "content": prompt}],
                    temperature=0.0
                )
                content = completion.choices[0].message.content
            elif self.provider == "gemini":
                model_instance = genai.GenerativeModel(self.model)
                response = model_instance.generate_content(prompt)
                content = response.text
            else:
                 return {"hallucination_score": 0.0, "hallucination_reasoning": "No valid API provider configured"}

            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                json_str = match.group(0)
                json_str = re.sub(r",\s*(?=[\]}])", "", json_str)
                data = json.loads(json_str)
                return {
                    "hallucination_score": data.get("score", 0.0),
                    "hallucination_reasoning": data.get("reasoning", "No reasoning provided")
                }
            else:
                 return {"hallucination_score": 0.0, "hallucination_reasoning": "Failed to parse LLM output"}

        except Exception as e:
            return {"hallucination_score": 0.0, "hallucination_reasoning": f"Error: {str(e)}"}
