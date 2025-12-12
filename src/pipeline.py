from typing import List
from .models import Conversation, Context, EvaluationResult
from .evaluators.base import BaseEvaluator
from .evaluators.relevance import RelevanceEvaluator
from .evaluators.hallucination import HallucinationEvaluator
from .evaluators.metrics import MetricsEvaluator

class EvaluationPipeline:
    def __init__(self):
        self.evaluators: List[BaseEvaluator] = [
            MetricsEvaluator(),
            RelevanceEvaluator(),
            HallucinationEvaluator()
        ]

    def run(self, conversation: Conversation, context: Context) -> EvaluationResult:
        results = {}
        for evaluator in self.evaluators:
            try:
                # Merge results from each evaluator
                results.update(evaluator.evaluate(conversation, context))
            except Exception as e:
                print(f"Evaluator {evaluator.__class__.__name__} failed: {e}")
        
        # map dict results to EvaluationResult model
        return EvaluationResult(
            relevance_score=results.get("relevance_score", 0.0),
            relevance_reasoning=results.get("relevance_reasoning", ""),
            hallucination_score=results.get("hallucination_score", 0.0),
            hallucination_reasoning=results.get("hallucination_reasoning", ""),
            latency_ms=results.get("latency_ms"),
            estimated_cost_usd=results.get("estimated_cost_usd", 0.0)
        )
