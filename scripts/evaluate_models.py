#!/usr/bin/env python3
"""
Evaluation framework for measuring model improvements.

Compares base model vs fine-tuned/RAG-enhanced versions.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from model evaluation."""
    model_name: str
    accuracy: float  # % of correct responses
    avg_latency_ms: float  # Response time
    perplexity: Optional[float]  # Language model quality
    bleu_score: Optional[float]  # Response similarity to reference
    user_satisfaction: Optional[float]  # If available


class ModelEvaluator:
    """
    Evaluate model performance on test dataset.
    """
    
    def __init__(self, test_data_path: Path):
        self.test_data = self._load_test_data(test_data_path)
    
    def _load_test_data(self, path: Path) -> list[dict]:
        """
        Load test conversations.
        
        Format:
        [
            {
                "user_message": "What's the weather?",
                "expected_intent": "weather_query",
                "reference_response": "Let me check the weather for you...",
                "quality_criteria": ["mentions_checking", "polite"]
            }
        ]
        """
        if not path.exists():
            logger.warning(f"Test data not found: {path}")
            return []
        
        with open(path) as f:
            return [json.loads(line) for line in f]
    
    async def evaluate_model(
        self,
        model_name: str,
        generate_response_fn,  # Function that takes user_message and returns response
    ) -> EvaluationResult:
        """
        Evaluate a model on test dataset.
        
        Args:
            model_name: Name for reporting
            generate_response_fn: Async function(user_message) -> response
        
        Returns:
            EvaluationResult with metrics
        """
        correct = 0
        total = len(self.test_data)
        latencies = []
        
        logger.info(f"Evaluating {model_name} on {total} examples...")
        
        for i, example in enumerate(self.test_data, 1):
            user_message = example["user_message"]
            expected_intent = example.get("expected_intent")
            quality_criteria = example.get("quality_criteria", [])
            
            # Measure latency
            import time
            start = time.time()
            response = await generate_response_fn(user_message)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
            
            # Check correctness
            is_correct = self._evaluate_response(
                response=response,
                expected_intent=expected_intent,
                quality_criteria=quality_criteria,
            )
            
            if is_correct:
                correct += 1
            
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{total} ({correct/i*100:.1f}% correct)")
        
        accuracy = correct / total if total > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        result = EvaluationResult(
            model_name=model_name,
            accuracy=accuracy,
            avg_latency_ms=avg_latency,
            perplexity=None,  # Would need to calculate
            bleu_score=None,  # Would need to calculate
            user_satisfaction=None,
        )
        
        logger.info(f"\n=== {model_name} Results ===")
        logger.info(f"Accuracy: {accuracy*100:.1f}%")
        logger.info(f"Avg Latency: {avg_latency:.1f}ms")
        
        return result
    
    def _evaluate_response(
        self,
        response: str,
        expected_intent: Optional[str],
        quality_criteria: list[str],
    ) -> bool:
        """
        Evaluate if response meets quality criteria.
        
        You can customize this based on your needs:
        - Intent detection accuracy
        - Response completeness
        - Politeness/tone
        - Factual correctness
        """
        # Simple keyword-based evaluation
        response_lower = response.lower()
        
        # Check quality criteria
        for criterion in quality_criteria:
            if criterion == "mentions_checking":
                if "check" not in response_lower:
                    return False
            elif criterion == "polite":
                polite_words = ["please", "sure", "happy to", "let me"]
                if not any(word in response_lower for word in polite_words):
                    return False
            # Add more criteria as needed
        
        return True
    
    def compare_models(self, results: list[EvaluationResult]) -> None:
        """
        Compare multiple model results side-by-side.
        """
        logger.info("\n=== Model Comparison ===")
        logger.info(f"{'Model':<30} {'Accuracy':<12} {'Latency (ms)':<15}")
        logger.info("-" * 60)
        
        for result in results:
            logger.info(
                f"{result.model_name:<30} "
                f"{result.accuracy*100:>6.1f}%    "
                f"{result.avg_latency_ms:>8.1f}"
            )
        
        # Find best model
        best = max(results, key=lambda r: r.accuracy)
        logger.info(f"\nBest accuracy: {best.model_name} ({best.accuracy*100:.1f}%)")


async def example_evaluation():
    """
    Example: Compare base model vs fine-tuned model.
    """
    evaluator = ModelEvaluator(Path("data/test_conversations.jsonl"))
    
    # Define model response functions
    async def base_model_response(user_message: str) -> str:
        # Use your base model
        # return await llm_service.generate(user_message)
        return "Base model response"
    
    async def finetuned_model_response(user_message: str) -> str:
        # Use fine-tuned model
        # return await finetuned_llm.generate(user_message)
        return "Fine-tuned model response"
    
    async def rag_enhanced_response(user_message: str) -> str:
        # Use RAG-enhanced response
        # context = pattern_store.build_context(user_message)
        # return await llm_service.generate(user_message, context=context)
        return "RAG-enhanced response"
    
    # Evaluate all models
    base_result = await evaluator.evaluate_model("Base Model", base_model_response)
    ft_result = await evaluator.evaluate_model("Fine-tuned", finetuned_model_response)
    rag_result = await evaluator.evaluate_model("RAG Enhanced", rag_enhanced_response)
    
    # Compare
    evaluator.compare_models([base_result, ft_result, rag_result])


def create_test_dataset_from_conversations(
    input_path: Path,
    output_path: Path,
    test_ratio: float = 0.2,
) -> None:
    """
    Split conversation data into train/test sets.
    
    Args:
        input_path: Full conversation dataset
        output_path: Where to save test set
        test_ratio: Percentage for test set (0.2 = 20%)
    """
    import random
    
    # Load all conversations
    with open(input_path) as f:
        all_data = [json.loads(line) for line in f]
    
    # Shuffle and split
    random.shuffle(all_data)
    split_idx = int(len(all_data) * (1 - test_ratio))
    
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
    # Save splits
    train_path = input_path.parent / f"train_{input_path.name}"
    test_path = output_path
    
    with open(train_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    
    with open(test_path, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    
    logger.info(f"Split: {len(train_data)} train, {len(test_data)} test")


if __name__ == "__main__":
    # Create test/train split
    create_test_dataset_from_conversations(
        input_path=Path("data/training/training_data.jsonl"),
        output_path=Path("data/test_conversations.jsonl"),
        test_ratio=0.2,
    )
    
    # Run evaluation
    asyncio.run(example_evaluation())
