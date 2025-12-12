import datetime
from typing import List, Optional, Any, Literal
from pydantic import BaseModel, root_validator, Field

class Message(BaseModel):
    turn: Optional[int] = None
    role: str # "User" or "AI/Chatbot"
    message: str
    created_at: Optional[datetime.datetime] = None

    @root_validator(pre=True)
    def parse_timestamp(cls, values):
        ts = values.get("created_at")
        if isinstance(ts, str):
            try:
                values["created_at"] = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                pass
        return values

class Conversation(BaseModel):
    chat_id: Optional[int] = None
    user_id: Optional[int] = None
    conversation_turns: List[Message]
    model: str = "gpt-4" 

    @property
    def messages(self) -> List[Message]:
        return self.conversation_turns

class ContextVector(BaseModel):
    id: int
    text: Optional[str] = "" # Default to empty string if missing
    source_url: Optional[str] = None
    created_at: Optional[str] = None
    tokens: Optional[int] = None

class ContextData(BaseModel):
    vector_data: List[ContextVector]
    sources: Optional[dict] = None

class Context(BaseModel):
    status: Optional[str] = None
    data: Optional[ContextData] = None
    
    @property
    def context_chunks(self) -> List[str]:
        if self.data and self.data.vector_data:
            # Filter out empty text to avoid noise
            return [v.text for v in self.data.vector_data if v.text]
        return []

class EvaluationResult(BaseModel):
    relevance_score: float = Field(description="Score 0-1 indicating relevance to query")
    relevance_reasoning: str
    hallucination_score: float = Field(description="Score 0-1 usually, 0=hallucinated, 1=faithful")
    hallucination_reasoning: str
    latency_ms: Optional[float]
    estimated_cost_usd: float
