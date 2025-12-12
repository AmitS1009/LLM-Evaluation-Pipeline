from abc import ABC, abstractmethod
from ..models import Conversation, Context

class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, conversation: Conversation, context: Context) -> dict:
        """
        Evaluate the conversation and context.
        Returns a dictionary with specific metric names and values.
        """
        pass
