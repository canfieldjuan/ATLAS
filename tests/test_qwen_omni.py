#!/usr/bin/env python3
"""
Test Qwen2.5-Omni 3B for speech-to-speech processing.
"""

import os
import torch
import soundfile as sf

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def test_qwen_omni():
    print("=" * 60)
    print("Testing Qwen2.5-Omni 3B Speech-to-Speech")
    print("=" * 60)

    if torch.cuda.is_available():
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Free VRAM: {free_mem / 1024**3:.1f} GB")

    print("\nLoading model and processor...")

    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    model_name = "Qwen/Qwen2.5-Omni-3B"

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    print("Model loaded!")

    # Test: Generate speech output
    print("\n--- Test: Speech Generation ---")
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "Hello, my name is Atlas. How can I help you today?"}]}
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)

    with torch.no_grad():
        text_ids, audio = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
        )

    text_response = processor.batch_decode(text_ids, skip_special_tokens=True)
    print(f"Text: {text_response}")

    if audio is not None:
        audio_np = audio.reshape(-1).detach().cpu().numpy()
        sf.write("test_qwen_omni_output.wav", audio_np, samplerate=24000)
        print(f"Audio saved: test_qwen_omni_output.wav ({len(audio_np)/24000:.1f}s)")

    print("\nTest complete!")


if __name__ == "__main__":
    test_qwen_omni()
