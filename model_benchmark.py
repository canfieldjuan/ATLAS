import time
import requests
import statistics

# Ollama API endpoint
API_URL = "http://localhost:11434/api/chat"
TAGS_URL = "http://localhost:11434/api/tags"

# Test prompts
TEST_PROMPTS = [
    "Write a Python function to calculate fibonacci numbers recursively.",
    "Explain the difference between a list and a tuple in Python.",
    "Write a simple HTTP server in Python using the http.server module.",
]

def benchmark_model(model_name: str, num_runs: int = 3) -> dict:
    """Benchmark a model's inference speed."""
    times = []
    tokens_per_sec = []
    
    print(f"\n{'='*50}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*50}")
    
    for i, prompt in enumerate(TEST_PROMPTS):
        for run in range(num_runs):
            start = time.time()
            
            response = requests.post(API_URL, json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "num_predict": 256,
                    "temperature": 0.7,
                }
            })
            
            elapsed = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                
                # Ollama provides eval_count and eval_duration
                eval_count = data.get("eval_count", 0)
                eval_duration = data.get("eval_duration", 0)  # in nanoseconds
                
                if eval_duration > 0:
                    tps = eval_count / (eval_duration / 1e9)  # convert ns to seconds
                else:
                    tps = 0
                
                times.append(elapsed)
                tokens_per_sec.append(tps)
                
                print(f"  Prompt {i+1}, Run {run+1}: {elapsed:.2f}s, {tps:.1f} tok/s ({eval_count} tokens)")
            else:
                print(f"  ERROR: {response.status_code} - {response.text[:100]}")
    
    return {
        "model": model_name,
        "avg_time": statistics.mean(times) if times else 0,
        "avg_tps": statistics.mean(tokens_per_sec) if tokens_per_sec else 0,
        "min_time": min(times) if times else 0,
        "max_time": max(times) if times else 0,
    }

def main():
    print("\n" + "="*60)
    print("MODEL PERFORMANCE BENCHMARK - OLLAMA")
    print("="*60)
    
    # Get available models from Ollama
    try:
        models_resp = requests.get(TAGS_URL)
        if models_resp.status_code == 200:
            models_data = models_resp.json().get("models", [])
            available = [m["name"] for m in models_data]
            print(f"\nAvailable models: {available}")
        else:
            print("Could not fetch models list")
            available = []
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running (try: ollama serve)")
        return
    
    # Models to test - look for GPT and Qwen models
    models_to_test = []
    for model in available:
        model_lower = model.lower()
        if "gpt" in model_lower or "qwen" in model_lower:
            models_to_test.append(model)
    
    if not models_to_test:
        print("\nNo GPT/Qwen models found. Available models:")
        for m in available:
            print(f"  - {m}")
        print("\nPlease pull the models first:")
        print("  ollama pull <model-name>")
        return
    
    print(f"\nTesting models: {models_to_test}")
    
    results = []
    for model in models_to_test:
        try:
            result = benchmark_model(model, num_runs=2)
            results.append(result)
        except Exception as e:
            print(f"Error benchmarking {model}: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    print(f"{'Model':<50} {'Avg Time':>10} {'Avg TPS':>10}")
    print("-"*70)
    
    for r in sorted(results, key=lambda x: x["avg_tps"], reverse=True):
        print(f"{r['model']:<50} {r['avg_time']:>9.2f}s {r['avg_tps']:>9.1f}")
    
    if len(results) >= 2:
        fastest = max(results, key=lambda x: x["avg_tps"])
        print(f"\n🏆 Fastest: {fastest['model']} at {fastest['avg_tps']:.1f} tokens/sec")

if __name__ == "__main__":
    main()
