#!/usr/bin/env python3
"""
Benchmark cold start latency for Qwen3 235B on Fireworks.ai vs Together.ai
"""
import time
import os
from openai import OpenAI

# API Keys
FIREWORKS_API_KEY = "fw_KZtBLJ5EvtmYFBBTFAkHVs"
TOGETHER_API_KEY = "tgp_v1_9cu0FpqTEg9m_oWqicDu0ZYyNmEyTOfESCfmHFk5WnQ"

# Model to test
MODEL_NAME = "accounts/fireworks/models/gpt-oss-120b"  # Fireworks format
TOGETHER_MODEL = "openai/gpt-oss-120b"  # Together format

# Test prompt
TEST_PROMPT = "Explain the concept of recursion with a simple example."


def test_fireworks_cold_start():
    """Test Fireworks.ai cold start latency"""
    print("\n" + "="*60)
    print("🔥 TESTING FIREWORKS.AI")
    print("="*60)
    
    client = OpenAI(
        api_key=FIREWORKS_API_KEY,
        base_url="https://api.fireworks.ai/inference/v1"
    )
    
    results = []
    
    for run in range(3):
        print(f"\n📊 Run {run + 1}/3")
        start = time.time()
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=100,
                temperature=0.7
            )
            
            elapsed = time.time() - start
            
            # Extract response
            result_text = response.choices[0].message.content
            tokens = len(result_text.split())  # Rough token estimate
            
            print(f"⏱️  Total time: {elapsed:.2f}s")
            print(f"📝 Response: {result_text[:100]}...")
            print(f"🔢 Tokens: ~{tokens}")
            
            results.append({
                "run": run + 1,
                "time": elapsed,
                "tokens": tokens,
                "tps": tokens / elapsed if elapsed > 0 else 0
            })
            
            # Wait between runs to allow potential cold start
            if run < 2:
                print("⏳ Waiting 30s for potential cold start...")
                time.sleep(30)
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({"run": run + 1, "error": str(e)})
    
    return results


def test_together_cold_start():
    """Test Together.ai cold start latency"""
    if not TOGETHER_API_KEY:
        print("\n⚠️  TOGETHER_API_KEY not set. Skipping Together.ai test.")
        print("   Set it via: export TOGETHER_API_KEY=your_key")
        return []
    
    print("\n" + "="*60)
    print("🤝 TESTING TOGETHER.AI")
    print("="*60)
    
    client = OpenAI(
        api_key=TOGETHER_API_KEY,
        base_url="https://api.together.xyz/v1"
    )
    
    results = []
    
    for run in range(3):
        print(f"\n📊 Run {run + 1}/3")
        start = time.time()
        
        try:
            response = client.chat.completions.create(
                model=TOGETHER_MODEL,
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=100,
                temperature=0.7
            )
            
            elapsed = time.time() - start
            
            # Extract response
            result_text = response.choices[0].message.content
            tokens = len(result_text.split())  # Rough token estimate
            
            print(f"⏱️  Total time: {elapsed:.2f}s")
            print(f"📝 Response: {result_text[:100]}...")
            print(f"🔢 Tokens: ~{tokens}")
            
            results.append({
                "run": run + 1,
                "time": elapsed,
                "tokens": tokens,
                "tps": tokens / elapsed if elapsed > 0 else 0
            })
            
            # Wait between runs
            if run < 2:
                print("⏳ Waiting 30s for potential cold start...")
                time.sleep(30)
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({"run": run + 1, "error": str(e)})
    
    return results


def print_comparison(fireworks_results, together_results):
    """Print comparison table"""
    print("\n" + "="*60)
    print("📊 RESULTS COMPARISON")
    print("="*60)
    
    # Fireworks stats
    fw_times = [r["time"] for r in fireworks_results if "time" in r]
    if fw_times:
        print(f"\n🔥 FIREWORKS.AI:")
        print(f"   Run 1 (Cold): {fw_times[0]:.2f}s")
        if len(fw_times) > 1:
            print(f"   Run 2 (Warm): {fw_times[1]:.2f}s")
        if len(fw_times) > 2:
            print(f"   Run 3 (Warm): {fw_times[2]:.2f}s")
        print(f"   Average: {sum(fw_times)/len(fw_times):.2f}s")
        print(f"   Best: {min(fw_times):.2f}s")
    
    # Together stats
    tg_times = [r["time"] for r in together_results if "time" in r]
    if tg_times:
        print(f"\n🤝 TOGETHER.AI:")
        print(f"   Run 1 (Cold): {tg_times[0]:.2f}s")
        if len(tg_times) > 1:
            print(f"   Run 2 (Warm): {tg_times[1]:.2f}s")
        if len(tg_times) > 2:
            print(f"   Run 3 (Warm): {tg_times[2]:.2f}s")
        print(f"   Average: {sum(tg_times)/len(tg_times):.2f}s")
        print(f"   Best: {min(tg_times):.2f}s")
    
    # Winner
    if fw_times and tg_times:
        print(f"\n🏆 WINNER:")
        fw_avg = sum(fw_times) / len(fw_times)
        tg_avg = sum(tg_times) / len(tg_times)
        
        if fw_avg < tg_avg:
            diff = ((tg_avg - fw_avg) / tg_avg) * 100
            print(f"   Fireworks.ai is {diff:.1f}% faster on average")
        else:
            diff = ((fw_avg - tg_avg) / fw_avg) * 100
            print(f"   Together.ai is {diff:.1f}% faster on average")
        
        print(f"\n💰 COLD START IMPACT:")
        if len(fw_times) > 1:
            fw_coldstart_penalty = ((fw_times[0] - fw_times[1]) / fw_times[1]) * 100
            print(f"   Fireworks.ai: +{fw_coldstart_penalty:.1f}% on cold start")
        if len(tg_times) > 1:
            tg_coldstart_penalty = ((tg_times[0] - tg_times[1]) / tg_times[1]) * 100
            print(f"   Together.ai: +{tg_coldstart_penalty:.1f}% on cold start")


def main():
    print("🚀 Cloud Model Latency Benchmark")
    print(f"📝 Model: GPT-OSS 120B")
    print(f"🎯 Testing cold start vs warm start latency")
    
    # Test Fireworks
    fireworks_results = test_fireworks_cold_start()
    
    # Test Together
    together_results = test_together_cold_start()
    
    # Print comparison
    print_comparison(fireworks_results, together_results)
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
