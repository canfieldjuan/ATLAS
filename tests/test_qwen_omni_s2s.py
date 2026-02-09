#!/usr/bin/env python3
"""
Test Qwen2.5-Omni 3B full speech-to-speech.
"""

import os
import torch
import soundfile as sf
from qwen_omni_utils import process_mm_info

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def test_speech_to_speech():
    print("=" * 60)
    print("Testing Qwen2.5-Omni 3B: Speech-to-Speech")
    print("=" * 60)

    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    model_name = "Qwen/Qwen2.5-Omni-3B"

    print("Loading model...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    print("Model loaded!")

    # Use the audio we generated as input
    input_audio = "test_qwen_omni_output.wav"

    if not os.path.exists(input_audio):
        print(f"No input audio found at {input_audio}")
        print("Creating test audio...")
        # Generate a simple test
        import numpy as np
        sr = 16000
        duration = 2
        t = np.linspace(0, duration, sr * duration)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
        sf.write(input_audio, audio, sr)

    print(f"\nProcessing audio input: {input_audio}")

    # Speech-to-speech conversation
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": input_audio}
            ]
        }
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True
    )
    inputs = inputs.to(model.device)

    print("Generating response...")
    with torch.no_grad():
        text_ids, audio = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
        )

    text_response = processor.batch_decode(text_ids, skip_special_tokens=True)
    print(f"\nText response: {text_response}")

    if audio is not None:
        audio_np = audio.reshape(-1).detach().cpu().numpy()
        output_file = "test_qwen_s2s_output.wav"
        sf.write(output_file, audio_np, samplerate=24000)
        print(f"Audio response saved: {output_file} ({len(audio_np)/24000:.1f}s)")
    else:
        print("No audio generated")

    print("\n" + "=" * 60)
    print("Speech-to-Speech test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_speech_to_speech()
