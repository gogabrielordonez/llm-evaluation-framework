"""
LLM Evaluation Framework
Core module for benchmarking and comparing LLM performance across multiple dimensions.
"""

import time
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import statistics


class MetricType(Enum):
    """Types of evaluation metrics"""
    LATENCY = "latency"
    TOKEN_USAGE = "token_usage"
    COST = "cost"
    QUALITY = "quality"
    ACCURACY = "accuracy"


@dataclass
class EvaluationResult:
    """Container for a single evaluation result"""
    model: str
    prompt: str
    response: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        """Calculate throughput"""
        if self.latency_ms > 0:
            return (self.output_tokens / self.latency_ms) * 1000
        return 0.0


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark results for a model"""
    model: str
    total_runs: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_input_tokens: float
    avg_output_tokens: float
    total_cost_usd: float
    avg_quality_score: Optional[float]
    tokens_per_second: float

    def to_dict(self) -> Dict:
        return {
            "model": self.model,
            "total_runs": self.total_runs,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "avg_input_tokens": round(self.avg_input_tokens, 1),
            "avg_output_tokens": round(self.avg_output_tokens, 1),
            "total_cost_usd": round(self.total_cost_usd, 4),
            "avg_quality_score": round(self.avg_quality_score, 2) if self.avg_quality_score else None,
            "tokens_per_second": round(self.tokens_per_second, 1)
        }


class ModelPricing:
    """Pricing configuration for different models (per 1K tokens)"""

    PRICING = {
        # Claude models (Anthropic)
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-2": {"input": 0.008, "output": 0.024},

        # GPT models (OpenAI)
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},

        # AWS Bedrock pricing
        "anthropic.claude-v2": {"input": 0.008, "output": 0.024},
        "anthropic.claude-3-sonnet": {"input": 0.003, "output": 0.015},
    }

    @classmethod
    def calculate_cost(cls, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD for a given model and token counts"""
        pricing = cls.PRICING.get(model, {"input": 0.01, "output": 0.03})
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost


class LLMEvaluator:
    """
    Main evaluation framework for benchmarking LLM performance.
    Supports multiple models, custom metrics, and aggregated reporting.
    """

    def __init__(self):
        self.results: List[EvaluationResult] = []
        self.quality_evaluator: Optional[Callable] = None

    def set_quality_evaluator(self, evaluator: Callable[[str, str], float]):
        """
        Set a custom quality evaluation function.
        Function should take (prompt, response) and return a score 0-1.
        """
        self.quality_evaluator = evaluator

    def evaluate(
        self,
        model: str,
        prompt: str,
        llm_call: Callable[[str], Dict],
        runs: int = 1
    ) -> List[EvaluationResult]:
        """
        Evaluate a model's performance on a given prompt.

        Args:
            model: Model identifier (e.g., 'claude-3-sonnet', 'gpt-4')
            prompt: The prompt to evaluate
            llm_call: Function that takes prompt and returns
                      {"response": str, "input_tokens": int, "output_tokens": int}
            runs: Number of times to run the evaluation

        Returns:
            List of EvaluationResult objects
        """
        run_results = []

        for i in range(runs):
            start_time = time.perf_counter()

            try:
                result = llm_call(prompt)

                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000

                input_tokens = result.get("input_tokens", 0)
                output_tokens = result.get("output_tokens", 0)
                total_tokens = input_tokens + output_tokens

                cost = ModelPricing.calculate_cost(model, input_tokens, output_tokens)

                # Calculate quality if evaluator is set
                quality_score = None
                if self.quality_evaluator:
                    quality_score = self.quality_evaluator(prompt, result["response"])

                eval_result = EvaluationResult(
                    model=model,
                    prompt=prompt,
                    response=result["response"],
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    cost_usd=cost,
                    quality_score=quality_score,
                    metadata={"run": i + 1, "timestamp": time.time()}
                )

                run_results.append(eval_result)
                self.results.append(eval_result)

            except Exception as e:
                print(f"Error in run {i + 1}: {e}")
                continue

        return run_results

    def get_summary(self, model: Optional[str] = None) -> List[BenchmarkSummary]:
        """
        Get aggregated benchmark summaries.

        Args:
            model: Optional filter for specific model

        Returns:
            List of BenchmarkSummary objects
        """
        # Group results by model
        model_results: Dict[str, List[EvaluationResult]] = {}

        for result in self.results:
            if model and result.model != model:
                continue
            if result.model not in model_results:
                model_results[result.model] = []
            model_results[result.model].append(result)

        summaries = []

        for model_name, results in model_results.items():
            latencies = [r.latency_ms for r in results]
            input_tokens = [r.input_tokens for r in results]
            output_tokens = [r.output_tokens for r in results]
            costs = [r.cost_usd for r in results]
            quality_scores = [r.quality_score for r in results if r.quality_score is not None]
            tps = [r.tokens_per_second for r in results]

            summary = BenchmarkSummary(
                model=model_name,
                total_runs=len(results),
                avg_latency_ms=statistics.mean(latencies),
                p50_latency_ms=statistics.median(latencies),
                p95_latency_ms=sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0],
                avg_input_tokens=statistics.mean(input_tokens),
                avg_output_tokens=statistics.mean(output_tokens),
                total_cost_usd=sum(costs),
                avg_quality_score=statistics.mean(quality_scores) if quality_scores else None,
                tokens_per_second=statistics.mean(tps)
            )

            summaries.append(summary)

        return summaries

    def compare_models(self, models: List[str]) -> Dict[str, Any]:
        """
        Generate a comparison report between multiple models.
        """
        summaries = {s.model: s for s in self.get_summary() if s.model in models}

        if len(summaries) < 2:
            return {"error": "Need at least 2 models to compare"}

        # Find best performer in each category
        comparison = {
            "fastest": min(summaries.values(), key=lambda x: x.avg_latency_ms).model,
            "cheapest": min(summaries.values(), key=lambda x: x.total_cost_usd).model,
            "highest_throughput": max(summaries.values(), key=lambda x: x.tokens_per_second).model,
            "models": {name: s.to_dict() for name, s in summaries.items()}
        }

        # Add quality comparison if available
        quality_scores = {name: s.avg_quality_score for name, s in summaries.items()
                        if s.avg_quality_score is not None}
        if quality_scores:
            comparison["highest_quality"] = max(quality_scores, key=quality_scores.get)

        return comparison

    def export_results(self, filepath: str):
        """Export all results to JSON"""
        data = {
            "results": [
                {
                    "model": r.model,
                    "prompt": r.prompt[:100] + "..." if len(r.prompt) > 100 else r.prompt,
                    "latency_ms": r.latency_ms,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cost_usd": r.cost_usd,
                    "quality_score": r.quality_score,
                    "tokens_per_second": r.tokens_per_second
                }
                for r in self.results
            ],
            "summaries": [s.to_dict() for s in self.get_summary()]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def clear_results(self):
        """Clear all stored results"""
        self.results = []


# Quality evaluation helpers
def simple_length_evaluator(prompt: str, response: str) -> float:
    """Simple quality metric based on response length appropriateness"""
    # Ideal response is 100-500 chars for most prompts
    length = len(response)
    if length < 50:
        return 0.3
    elif length < 100:
        return 0.6
    elif length <= 500:
        return 1.0
    elif length <= 1000:
        return 0.8
    else:
        return 0.6


def keyword_coverage_evaluator(keywords: List[str]) -> Callable[[str, str], float]:
    """Create an evaluator that checks for keyword coverage in response"""
    def evaluator(prompt: str, response: str) -> float:
        response_lower = response.lower()
        found = sum(1 for kw in keywords if kw.lower() in response_lower)
        return found / len(keywords) if keywords else 0.0
    return evaluator


if __name__ == "__main__":
    # Demo usage
    evaluator = LLMEvaluator()
    evaluator.set_quality_evaluator(simple_length_evaluator)

    # Mock LLM call for testing
    def mock_llm(prompt: str) -> Dict:
        time.sleep(0.1)  # Simulate latency
        return {
            "response": f"This is a mock response to: {prompt[:50]}...",
            "input_tokens": len(prompt.split()) * 1.3,
            "output_tokens": 50
        }

    # Run evaluation
    results = evaluator.evaluate("claude-3-sonnet", "Explain quantum computing", mock_llm, runs=3)

    print("Evaluation Results:")
    for r in results:
        print(f"  Latency: {r.latency_ms:.2f}ms, Cost: ${r.cost_usd:.4f}")

    summary = evaluator.get_summary()[0]
    print(f"\nSummary for {summary.model}:")
    print(f"  Avg Latency: {summary.avg_latency_ms:.2f}ms")
    print(f"  Total Cost: ${summary.total_cost_usd:.4f}")
