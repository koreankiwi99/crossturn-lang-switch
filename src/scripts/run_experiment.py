#!/usr/bin/env python3
"""
Run Experiment - Generate Model Responses (Parallel Version)

Send multi-turn conversations to models and save responses.
Works with both original MultiChallenge and code-switching datasets.

Usage:
    # Original MultiChallenge
    python run_experiment.py --model gemini-3-pro --workers 32
    python run_experiment.py --model gpt-4o --samples 20

    # Code-switching dataset
    python run_experiment.py --model gemini-3-pro --data data/codeswitching/en_to_x/codeswitching_de.jsonl --workers 32
"""

import os
import sys
import json
import argparse
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from tqdm import tqdm

load_dotenv()


# === MODEL CONFIGURATIONS ===
MODELS = {
    # Apertus (EPFL) - Multilingual, 1000+ languages
    "apertus-70b": {
        "provider": "apertus",
        "model_id": "swiss-ai/Apertus-70B-Instruct-2509",
        "base_url": "https://inference.rcp.epfl.ch/v1",
        "api_key_env": "APERTUS_API_KEY",
    },
    "apertus-8b": {
        "provider": "apertus",
        "model_id": "swiss-ai/Apertus-8B-Instruct-2509",
        "base_url": "https://inference.rcp.epfl.ch/v1",
        "api_key_env": "APERTUS_API_KEY",
    },
    # OpenAI
    "gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o-2024-08-06",
        "api_key_env": "OPENAI_API_KEY",
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "gpt-4.1": {
        "provider": "openai",
        "model_id": "gpt-4.1-2025-04-14",
        "api_key_env": "OPENAI_API_KEY",
    },
    "gpt-5": {
        "provider": "openai",
        "model_id": "gpt-5-2025-08-07",
        "api_key_env": "OPENAI_API_KEY",
        "skip_max_tokens": True,  # GPT-5 doesn't support max_tokens
        "skip_temperature": True,  # GPT-5 doesn't support temperature
    },
    # Anthropic (direct)
    "claude-opus-4.5": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-5-20251101",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    # OpenRouter
    "gemini-2.5-pro": {
        "provider": "openrouter",
        "model_id": "google/gemini-2.5-pro-preview",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "gemini-3-pro": {
        "provider": "openrouter",
        "model_id": "google/gemini-3-pro-preview",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "llama-4-maverick": {
        "provider": "openrouter",
        "model_id": "meta-llama/llama-4-maverick",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "qwen3-235b": {
        "provider": "openrouter",
        "model_id": "qwen/qwen3-235b-a22b",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "temperature": 0.7,  # Qwen3 recommends non-zero temp; 0.7 for non-thinking mode
    },
    # Claude via OpenRouter
    "claude-opus-4.5-openrouter": {
        "provider": "openrouter",
        "model_id": "anthropic/claude-opus-4.5",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    # DeepSeek via OpenRouter
    "deepseek-v3.1": {
        "provider": "openrouter",
        "model_id": "deepseek/deepseek-chat-v3.1",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "temperature": 0.0,  # Match MultiChallenge leaderboard evaluation (non-thinking mode)
    },
    # Cohere via OpenRouter
    "command-r-plus": {
        "provider": "openrouter",
        "model_id": "cohere/command-r-plus-08-2024",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "temperature": 0.0,  # Cohere recommends temp=0 for deterministic outputs
    },
}


def get_client(model_config):
    """Get client for model (OpenAI-compatible or Anthropic)."""
    api_key = os.getenv(model_config["api_key_env"])
    if not api_key:
        raise ValueError(f"{model_config['api_key_env']} not set in environment")

    provider = model_config.get("provider")

    if provider == "anthropic":
        return anthropic.Anthropic(api_key=api_key)
    elif provider == "openrouter":
        return OpenAI(
            base_url=model_config["base_url"],
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/codeswitching-research",
                "X-Title": "Code-Switching Research"
            }
        )
    elif "base_url" in model_config:
        return OpenAI(base_url=model_config["base_url"], api_key=api_key)
    else:
        return OpenAI(api_key=api_key)


def load_dataset(data_path, axes=None):
    """Load dataset (MultiChallenge or code-switching)."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]

    if axes:
        data = [d for d in data if d['AXIS'] in axes]

    # Add turn count
    for item in data:
        item['turn_count'] = len(item['CONVERSATION'])

    return data


def query_model(client, model_id, messages, provider, max_tokens=4096, temperature=0.0, model_config=None, max_retries=3):
    """Send conversation to model, get response. Retries on empty responses."""
    import time

    # Use model-specific temperature if configured
    if model_config and "temperature" in model_config:
        temperature = model_config["temperature"]

    for attempt in range(max_retries):
        try:
            if provider == "anthropic":
                response = client.messages.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                result = {
                    "success": True,
                    "response": response.content[0].text if response.content else "",
                    "finish_reason": response.stop_reason,
                    "usage": {
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                    }
                }
            else:
                # Build kwargs, optionally skipping max_tokens/temperature for models that don't support them
                kwargs = {
                    "model": model_id,
                    "messages": messages,
                }
                if not (model_config and model_config.get("skip_temperature")):
                    kwargs["temperature"] = temperature
                if not (model_config and model_config.get("skip_max_tokens")):
                    kwargs["max_tokens"] = max_tokens
                response = client.chat.completions.create(**kwargs)
                result = {
                    "success": True,
                    "response": response.choices[0].message.content or "",
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    }
                }

            # Retry if response is empty (transient API issue)
            if result["response"]:
                return result
            elif attempt < max_retries - 1:
                time.sleep(1)  # Brief pause before retry
                continue
            else:
                result["retry_exhausted"] = True
                return result

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return {"success": False, "error": str(e)}


def process_single_item(args):
    """Process a single item (for parallel execution)."""
    item, client, model_id, provider, model_config, model_name = args

    response_data = query_model(client, model_id, item['CONVERSATION'], provider, model_config=model_config)

    result = {
        "question_id": item["QUESTION_ID"],
        "axis": item["AXIS"],
        "turn_count": item["turn_count"],
        "target_question": item["TARGET_QUESTION"],
        "pass_criteria": item["PASS_CRITERIA"],
        "conversation": item["CONVERSATION"],
        "model": model_name,
        "model_id": model_id,
        "timestamp": datetime.now().isoformat(),
        **response_data
    }

    # Include code-switch info if present
    if "CODE_SWITCH" in item:
        result["code_switch"] = item["CODE_SWITCH"]

    return result


def run_experiment_parallel(data, client, model_name, model_id, provider, output_dir,
                           num_workers=32, resume_file=None, model_config=None, dataset_name=None):
    """Generate responses for all questions using parallel workers."""
    os.makedirs(output_dir, exist_ok=True)

    # Resume or create new
    if resume_file:
        output_file = resume_file
        completed_ids = set()
        with open(resume_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    if result.get("success"):
                        completed_ids.add(result["question_id"])
        print(f"\nResuming from {resume_file}")
        print(f"Already completed: {len(completed_ids)} questions")
        data = [d for d in data if d["QUESTION_ID"] not in completed_ids]
        print(f"Remaining: {len(data)} questions")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_suffix = f"_{dataset_name}" if dataset_name else ""
        output_file = f"{output_dir}/responses{name_suffix}_{timestamp}.jsonl"

    print(f"\nGenerating responses with {model_name}...")
    print(f"Workers: {num_workers}")
    print(f"Output: {output_file}")
    print("=" * 60)

    stats = {"total": len(data), "success": 0, "error": 0}
    file_lock = threading.Lock()

    # Prepare arguments for parallel processing
    task_args = [
        (item, client, model_id, provider, model_config, model_name)
        for item in data
    ]

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_item, args): args[0]["QUESTION_ID"]
                   for args in task_args}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating responses"):
            result = future.result()
            results.append(result)

            if result.get("success"):
                stats["success"] += 1
            else:
                stats["error"] += 1
                tqdm.write(f"ERROR: {result.get('error', 'Unknown')[:80]}")

            # Write immediately (thread-safe)
            with file_lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f"COMPLETE: {stats['success']}/{stats['total']} successful")
    print(f"Responses saved: {output_file}")
    print("=" * 60)
    print(f"\nNext step - Evaluate:")
    print(f"  python evaluate.py --input {output_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Run experiment - generate responses")

    parser.add_argument("--model", type=str, required=True,
                       choices=list(MODELS.keys()),
                       help="Model to test")

    parser.add_argument("--data", type=str, default=None,
                       help="Path to dataset (default: MultiChallenge)")

    parser.add_argument("--samples", type=int, default=None,
                       help="Number of samples (default: all)")

    parser.add_argument("--axis", type=str, default=None, nargs="+",
                       choices=["INFERENCE_MEMORY", "INSTRUCTION_RETENTION",
                               "SELF_COHERENCE", "RELIABLE_VERSION_EDITING"],
                       help="Filter by axis")

    parser.add_argument("--resume", type=str, default=None,
                       help="Path to existing results file to resume from")

    parser.add_argument("--workers", type=int, default=32,
                       help="Number of parallel workers (default: 32)")

    args = parser.parse_args()

    model_config = MODELS[args.model]

    print("=" * 60)
    print("EXPERIMENT: Generate Responses (Parallel)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Model ID: {model_config['model_id']}")

    # Data path
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if args.data:
        data_path = args.data if os.path.isabs(args.data) else os.path.join(base_dir, args.data)
    else:
        data_path = os.path.join(base_dir, "data/multi-challenge/data/benchmark_questions.jsonl")

    # Load data
    axes = args.axis or ["INFERENCE_MEMORY", "INSTRUCTION_RETENTION"]
    data = load_dataset(data_path, axes=axes if not args.data else None)
    print(f"Dataset: {len(data)} questions from {data_path}")

    if args.samples and len(data) > args.samples:
        data = data[:args.samples]
        print(f"Using first {args.samples} samples")

    # Client
    client = get_client(model_config)

    # Output directory
    output_dir = os.path.join(base_dir, f"results/{args.model}")

    # Extract dataset name from path for output file naming
    dataset_name = None
    if args.data:
        # e.g., "codeswitching_de" from "data/codeswitching/codeswitching_de.jsonl"
        dataset_name = os.path.splitext(os.path.basename(args.data))[0]

    # Run
    provider = model_config.get('provider', 'openai')
    run_experiment_parallel(data, client, args.model, model_config['model_id'], provider,
                           output_dir, num_workers=args.workers, resume_file=args.resume,
                           model_config=model_config, dataset_name=dataset_name)


if __name__ == "__main__":
    main()
