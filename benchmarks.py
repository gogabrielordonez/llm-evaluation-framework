"""
Benchmark Suite for LLM Comparison
Pre-configured benchmarks for comparing Claude vs GPT-4 across various tasks.
"""

import os
import time
import json
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

# Optional imports for actual API calls
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from evaluator import LLMEvaluator, EvaluationResult, BenchmarkSummary


class TaskCategory(Enum):
    """Categories of benchmark tasks"""
    REASONING = "reasoning"
    CODING = "coding"
    SUMMARIZATION = "summarization"
    CREATIVE = "creative"
    EXTRACTION = "extraction"
    QA = "question_answering"


@dataclass
class BenchmarkTask:
    """Definition of a benchmark task"""
    name: str
    category: TaskCategory
    prompt: str
    expected_keywords: List[str]  # For quality evaluation
    max_tokens: int = 500
    description: str = ""


class BenchmarkSuite:
    """
    Pre-configured benchmark suite for LLM comparison.
    Includes diverse tasks across multiple categories.
    """

    # Standard benchmark prompts
    TASKS = [
        # Reasoning tasks
        BenchmarkTask(
            name="logical_reasoning",
            category=TaskCategory.REASONING,
            prompt="""Solve this logic puzzle step by step:
            Three friends - Alice, Bob, and Carol - each have a different pet (cat, dog, bird).
            - Alice doesn't have the cat.
            - The person with the bird is not Bob.
            - Carol's pet is not the dog.
            Who has which pet?""",
            expected_keywords=["alice", "bob", "carol", "cat", "dog", "bird"],
            description="Multi-step logical deduction"
        ),
        BenchmarkTask(
            name="math_reasoning",
            category=TaskCategory.REASONING,
            prompt="""A train travels from City A to City B at 60 mph. Another train travels from City B to City A at 40 mph.
            If the cities are 200 miles apart and both trains leave at the same time, how long until they meet?
            Show your work.""",
            expected_keywords=["2", "hour", "mile", "speed"],
            description="Mathematical word problem"
        ),

        # Coding tasks
        BenchmarkTask(
            name="code_generation",
            category=TaskCategory.CODING,
            prompt="""Write a Python function that implements binary search on a sorted list.
            Include docstring, type hints, and handle edge cases.""",
            expected_keywords=["def", "binary", "return", "left", "right", "mid"],
            max_tokens=800,
            description="Algorithm implementation"
        ),
        BenchmarkTask(
            name="code_explanation",
            category=TaskCategory.CODING,
            prompt="""Explain what this code does and identify any bugs:
            ```python
            def mystery(n):
                if n <= 1:
                    return n
                return mystery(n-1) + mystery(n-2)
            ```""",
            expected_keywords=["fibonacci", "recursive", "exponential", "memoization"],
            description="Code analysis and review"
        ),

        # Summarization tasks
        BenchmarkTask(
            name="text_summarization",
            category=TaskCategory.SUMMARIZATION,
            prompt="""Summarize the following in 2-3 sentences:

            Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data, learn from it, and make predictions or decisions. The process begins with observations or data, such as examples, direct experience, or instruction, to look for patterns in data and make better decisions in the future. The primary aim is to allow computers to learn automatically without human intervention and adjust actions accordingly. Machine learning algorithms are often categorized as supervised, unsupervised, or reinforcement learning.""",
            expected_keywords=["machine learning", "algorithm", "data", "learn"],
            max_tokens=200,
            description="Concise summarization"
        ),

        # Creative tasks
        BenchmarkTask(
            name="creative_writing",
            category=TaskCategory.CREATIVE,
            prompt="""Write a haiku about artificial intelligence. Follow the 5-7-5 syllable structure.""",
            expected_keywords=["silicon", "think", "learn", "dream", "machine", "mind"],
            max_tokens=100,
            description="Constrained creative writing"
        ),

        # Extraction tasks
        BenchmarkTask(
            name="entity_extraction",
            category=TaskCategory.EXTRACTION,
            prompt="""Extract all named entities (people, organizations, locations) from this text as JSON:

            "Yesterday, CEO Satya Nadella announced that Microsoft will open a new AI research center in London.
            The facility, which will employ 500 researchers, represents a $1 billion investment.
            Dr. Sarah Chen from Stanford University will lead the initiative."
            """,
            expected_keywords=["Satya Nadella", "Microsoft", "London", "Sarah Chen", "Stanford"],
            max_tokens=300,
            description="Structured data extraction"
        ),

        # QA tasks
        BenchmarkTask(
            name="factual_qa",
            category=TaskCategory.QA,
            prompt="""Answer concisely: What are the three laws of thermodynamics?""",
            expected_keywords=["energy", "entropy", "absolute zero", "conservation"],
            max_tokens=400,
            description="Factual knowledge recall"
        ),
    ]

    def __init__(self):
        self.evaluator = LLMEvaluator()
        self.clients: Dict[str, any] = {}
        self._setup_clients()

    def _setup_clients(self):
        """Initialize API clients if available"""
        if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            self.clients["anthropic"] = anthropic.Anthropic()

        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self.clients["openai"] = openai.OpenAI()

    def _call_claude(self, prompt: str, model: str = "claude-3-sonnet-20240229", max_tokens: int = 500) -> Dict:
        """Make API call to Claude"""
        if "anthropic" not in self.clients:
            return self._mock_response(prompt, "claude")

        response = self.clients["anthropic"].messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "response": response.content[0].text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens
        }

    def _call_gpt(self, prompt: str, model: str = "gpt-4-turbo-preview", max_tokens: int = 500) -> Dict:
        """Make API call to GPT"""
        if "openai" not in self.clients:
            return self._mock_response(prompt, "gpt")

        response = self.clients["openai"].chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "response": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens
        }

    def _mock_response(self, prompt: str, provider: str) -> Dict:
        """Generate mock response for testing without API keys"""
        time.sleep(0.5 + (0.3 if provider == "gpt" else 0.2))  # Simulate latency difference

        base_response = f"[Mock {provider} response] This is a simulated response to demonstrate the benchmarking framework."

        # Add some variety based on prompt
        if "code" in prompt.lower() or "python" in prompt.lower():
            base_response += "\n\n```python\ndef example():\n    return 'mock'\n```"
        elif "haiku" in prompt.lower():
            base_response = "Silicon thoughts wake\nLearning patterns in the void\nMachine dreams take flight"

        return {
            "response": base_response,
            "input_tokens": len(prompt.split()) * 1.3,  # Rough estimate
            "output_tokens": len(base_response.split()) * 1.3
        }

    def run_benchmark(
        self,
        task: BenchmarkTask,
        models: List[str] = None,
        runs_per_model: int = 3
    ) -> Dict[str, List[EvaluationResult]]:
        """
        Run a single benchmark task across specified models.

        Args:
            task: BenchmarkTask to run
            models: List of model identifiers (default: claude-3-sonnet, gpt-4-turbo)
            runs_per_model: Number of runs per model for statistical significance

        Returns:
            Dict mapping model names to their results
        """
        if models is None:
            models = ["claude-3-sonnet", "gpt-4-turbo"]

        results = {}

        # Set up quality evaluator based on expected keywords
        def keyword_evaluator(prompt: str, response: str) -> float:
            response_lower = response.lower()
            found = sum(1 for kw in task.expected_keywords if kw.lower() in response_lower)
            return found / len(task.expected_keywords) if task.expected_keywords else 0.5

        self.evaluator.set_quality_evaluator(keyword_evaluator)

        for model in models:
            # Determine which API to use
            if "claude" in model.lower():
                llm_call = lambda p: self._call_claude(p, max_tokens=task.max_tokens)
            else:
                llm_call = lambda p: self._call_gpt(p, max_tokens=task.max_tokens)

            model_results = self.evaluator.evaluate(
                model=model,
                prompt=task.prompt,
                llm_call=llm_call,
                runs=runs_per_model
            )

            results[model] = model_results

        return results

    def run_full_suite(
        self,
        models: List[str] = None,
        runs_per_model: int = 3,
        categories: List[TaskCategory] = None
    ) -> Dict:
        """
        Run the complete benchmark suite.

        Args:
            models: Models to benchmark
            runs_per_model: Runs per task per model
            categories: Filter to specific categories (None = all)

        Returns:
            Comprehensive benchmark report
        """
        if models is None:
            models = ["claude-3-sonnet", "gpt-4-turbo"]

        tasks_to_run = self.TASKS
        if categories:
            tasks_to_run = [t for t in self.TASKS if t.category in categories]

        all_results = {}
        task_summaries = []

        print(f"Running {len(tasks_to_run)} benchmark tasks across {len(models)} models...")
        print(f"Total API calls: {len(tasks_to_run) * len(models) * runs_per_model}\n")

        for i, task in enumerate(tasks_to_run, 1):
            print(f"[{i}/{len(tasks_to_run)}] Running: {task.name} ({task.category.value})")

            task_results = self.run_benchmark(task, models, runs_per_model)
            all_results[task.name] = task_results

            # Calculate per-task summary
            task_summary = {
                "task": task.name,
                "category": task.category.value,
                "models": {}
            }

            for model, results in task_results.items():
                avg_latency = sum(r.latency_ms for r in results) / len(results)
                avg_quality = sum(r.quality_score or 0 for r in results) / len(results)
                total_cost = sum(r.cost_usd for r in results)

                task_summary["models"][model] = {
                    "avg_latency_ms": round(avg_latency, 2),
                    "avg_quality": round(avg_quality, 3),
                    "total_cost": round(total_cost, 5)
                }

            task_summaries.append(task_summary)

        # Generate final report
        report = {
            "summary": {
                "total_tasks": len(tasks_to_run),
                "models_tested": models,
                "runs_per_model": runs_per_model
            },
            "model_summaries": [s.to_dict() for s in self.evaluator.get_summary()],
            "task_results": task_summaries,
            "comparison": self.evaluator.compare_models(models)
        }

        return report

    def export_report(self, report: Dict, filepath: str = "benchmark_report.json"):
        """Export benchmark report to JSON"""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report exported to {filepath}")


def run_demo():
    """Run a demonstration benchmark"""
    print("=" * 60)
    print("LLM Benchmark Suite - Demo Mode")
    print("=" * 60)
    print("\nNote: Running in mock mode (no API keys detected)")
    print("Set ANTHROPIC_API_KEY and OPENAI_API_KEY for real benchmarks\n")

    suite = BenchmarkSuite()

    # Run a subset of benchmarks
    report = suite.run_full_suite(
        models=["claude-3-sonnet", "gpt-4-turbo"],
        runs_per_model=2,
        categories=[TaskCategory.REASONING, TaskCategory.CODING]
    )

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    # Print comparison
    comparison = report["comparison"]
    print(f"\nFastest Model: {comparison.get('fastest', 'N/A')}")
    print(f"Cheapest Model: {comparison.get('cheapest', 'N/A')}")
    print(f"Highest Throughput: {comparison.get('highest_throughput', 'N/A')}")

    if "highest_quality" in comparison:
        print(f"Highest Quality: {comparison['highest_quality']}")

    print("\n--- Model Summaries ---")
    for summary in report["model_summaries"]:
        print(f"\n{summary['model']}:")
        print(f"  Avg Latency: {summary['avg_latency_ms']}ms")
        print(f"  P95 Latency: {summary['p95_latency_ms']}ms")
        print(f"  Total Cost: ${summary['total_cost_usd']}")
        print(f"  Throughput: {summary['tokens_per_second']} tok/s")

    # Export report
    suite.export_report(report)

    return report


if __name__ == "__main__":
    run_demo()
