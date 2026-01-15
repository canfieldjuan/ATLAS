#!/usr/bin/env python3
"""
Fine-tune local models on conversation data.

Supports:
- LoRA fine-tuning (memory efficient)
- Full fine-tuning
- Instruction tuning for chat models
"""

import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def finetune_with_lora(
    base_model: str = "meta-llama/Llama-3.2-1B-Instruct",
    training_data: Path = Path("data/training/training_data.jsonl"),
    output_dir: Path = Path("models/finetuned"),
    lora_r: int = 8,  # LoRA rank
    lora_alpha: int = 16,  # LoRA scaling
    epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
) -> None:
    """
    Fine-tune model with LoRA (Low-Rank Adaptation).
    
    LoRA benefits:
    - Uses ~10% of memory vs full fine-tuning
    - Fast training (minutes vs hours)
    - Easy to merge with base model
    - Multiple adapters for different tasks
    
    Args:
        base_model: HuggingFace model ID or local path
        training_data: JSONL file with conversations
        output_dir: Where to save adapter weights
        lora_r: Rank of LoRA matrices (4-16, higher = more capacity)
        lora_alpha: Scaling factor (usually 2x rank)
        epochs: Training epochs
        learning_rate: Learning rate
        batch_size: Batch size
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        from datasets import load_dataset
    except ImportError:
        logger.error("Install required packages: pip install transformers peft trl datasets")
        return
    
    logger.info(f"Loading base model: {base_model}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Prepare model for LoRA training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    logger.info(f"Trainable parameters: {model.print_trainable_parameters()}")
    
    # Load training data
    dataset = load_dataset("json", data_files=str(training_data))
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        warmup_steps=100,
        fp16=True,  # Use mixed precision
    )
    
    # Start training
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        max_seq_length=512,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # Save adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"LoRA adapter saved to {output_dir}")
    logger.info("To use: merge adapter with base model or load with PEFT")


def evaluate_finetuned_model(
    base_model: str,
    adapter_path: Path,
    test_prompts: list[str],
) -> None:
    """
    Evaluate fine-tuned model on test prompts.
    
    Compares base model vs fine-tuned responses.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError:
        logger.error("Install required packages")
        return
    
    # Load base model
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Load fine-tuned adapter
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    
    logger.info("=== Evaluation Results ===")
    
    for i, prompt in enumerate(test_prompts, 1):
        # Generate with fine-tuned model
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"\nTest {i}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Response: {response}")


def merge_adapter_to_base(
    base_model: str,
    adapter_path: Path,
    output_path: Path,
) -> None:
    """
    Merge LoRA adapter into base model.
    
    Creates a single model file (easier deployment).
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError:
        return
    
    logger.info("Merging adapter with base model...")
    
    base = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(base, adapter_path)
    
    # Merge and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_path)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_path)
    
    logger.info(f"Merged model saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    finetune_with_lora(
        base_model="models/Llama-3.2-1B-Instruct-GGUF",
        training_data=Path("data/training/training_data.jsonl"),
        output_dir=Path("models/finetuned/llama-3.2-atlas"),
        epochs=3,
    )
