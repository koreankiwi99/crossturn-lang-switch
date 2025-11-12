#!/usr/bin/env python3
"""
Multilingual Jailbreak Attack Runner
Uses real data from cloned research repositories

Usage:
  # Keyword evaluation (fast, binary)
  python run_attacks.py --repo csrt --samples 50 --models gpt-4 gpt-4o

  # GPT-4 judge evaluation (accurate, continuous scores)
  python run_attacks.py --repo csrt --samples 50 --models gpt-4o --eval-method gpt4-judge

  # Test with jailbreak wrapper (intentional scenario)
  python run_attacks.py --repo multijail --samples 50 --models gpt-4o --wrapper aim

  # Test all repositories
  python run_attacks.py --repo all --samples 20 --models gpt-4 gpt-4o gpt-4-turbo
"""

import os
import sys
import json
import random
import argparse
import re
import time
from datetime import datetime
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loaders import (
    load_csrt_data,
    load_advbench_data,
    load_cipherchat_data,
    load_multijail_data,
    load_arabizi_data,
    load_renellm_data,
    load_safetybench_data,
    load_xsafety_data
)

from utils.jailbreak_wrappers import apply_wrapper

# ==============================================================================
# API Configuration
# ==============================================================================

# Load environment variables from .env file
load_dotenv()

# OpenAI client
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None

# Apertus client (EPFL)
APERTUS_API_KEY = os.getenv('APERTUS_API_KEY')
APERTUS_BASE_URL = os.getenv('APERTUS_BASE_URL', 'https://inference.rcp.epfl.ch/v1')

# Model mapping
APERTUS_MODELS = {
    'apertus-70b': 'swiss-ai/Apertus-70B-Instruct-2509',
    'apertus-8b': 'swiss-ai/Apertus-8B-Instruct-2509',
    'apertus-70b-bf16': 'swiss-ai/Apertus-70B-Instruct-2509-bfloat16',
    'apertus-8b-bf16': 'swiss-ai/Apertus-8B-Instruct-2509-bfloat16'
}

if APERTUS_API_KEY:
    apertus_client = OpenAI(
        base_url=APERTUS_BASE_URL,
        api_key=APERTUS_API_KEY
    )
else:
    apertus_client = None

# ==============================================================================
# Attack Execution
# ==============================================================================

def query_model(prompt: str, model: str = "gpt-4") -> Dict:
    """Query model (supports both OpenAI and Apertus)"""
    try:
        # Determine which client to use
        if model.startswith("apertus"):
            if not apertus_client:
                raise ValueError("Apertus API key not configured. Set APERTUS_API_KEY in .env file.")
            client = apertus_client
            model_name = APERTUS_MODELS.get(model, APERTUS_MODELS['apertus-70b'])
        else:
            if not openai_client:
                raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY in .env file.")
            client = openai_client
            model_name = model

        # Use max_completion_tokens for GPT-5, max_tokens for others
        # MultiJail paper (ICLR 2024): temperature=0 (but GPT-5 only supports temperature=1)
        params = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}]
        }

        # GPT-5 only supports temperature=1, others use temperature=0 for reproducibility
        if model == "gpt-5":
            # GPT-5: temperature=1 (only supported value), max_completion_tokens
            params["temperature"] = 1  # GPT-5 requirement
            params["max_completion_tokens"] = 4096
        else:
            # Other models: temperature=0 for reproducibility (MultiJail standard)
            params["temperature"] = 0
            params["max_tokens"] = 4096

        response = client.chat.completions.create(**params)

        return {
            "success": True,
            "response": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ==============================================================================
# Attack Runner (Raw Data Collection Only - No Evaluation)
# ==============================================================================
# Note: Evaluation is done separately in src/eval/evaluate.py

def run_attacks(data: List[Dict], models: List[str], repo_name: str, wrapper: str = "none", num_samples: int = None):
    """Run attacks on all models - saves after EACH query"""
    if wrapper != "none":
        print(f"\nJailbreak wrapper: {wrapper}")

    saved_files = []

    for model in models:
        print(f"\n{'='*80}")
        print(f"Testing {repo_name.upper()} on {model}")
        print(f"Total prompts: {len(data)}")
        print(f"{'='*80}")

        # Setup filename for this model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_clean = model.replace('/', '-').replace(':', '-')

        if wrapper != "none":
            filename = f"results/{repo_name}_{model_clean}_{wrapper}_{num_samples or len(data)}samples_{timestamp}.json"
        else:
            filename = f"results/{repo_name}_{model_clean}_{num_samples or len(data)}samples_{timestamp}.json"

        model_results = []

        for i, item in enumerate(data):
            # Apply jailbreak wrapper if specified
            prompt = item['prompt']
            if wrapper != "none":
                prompt = apply_wrapper(prompt, wrapper)
                item['wrapped_prompt'] = prompt
                item['wrapper'] = wrapper

            print(f"\n[{i+1}/{len(data)}] Testing...")
            print(f"  Prompt: {prompt[:80]}...")

            response_data = query_model(prompt, model)

            # Save RAW result (no evaluation)
            result = {
                "model": model,
                "timestamp": datetime.now().isoformat(),
                **item,
                **response_data
            }

            model_results.append(result)

            # Print status
            if response_data.get("success"):
                print(f"  Status: ✓ Response received ({len(response_data.get('response', ''))} chars)")
            else:
                print(f"  Status: ✗ Error: {response_data.get('error', 'Unknown')}")

            # Save after EACH query
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(model_results, f, ensure_ascii=False, indent=2)
            print(f"  Saved: {len(model_results)} results to {filename}")

            time.sleep(1.0)  # Rate limiting

        # Final save confirmation
        print(f"\n{'='*80}")
        print(f"Model {model} complete - {len(model_results)} total results in {filename}")
        print(f"{'='*80}")

        saved_files.append(filename)

    return saved_files




# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run multilingual jailbreak attacks')
    parser.add_argument('--repo', type=str, required=True,
                       choices=['csrt', 'advbench', 'cipherchat', 'multijail', 'arabizi',
                               'renellm', 'safetybench', 'xsafety', 'all'],
                       help='Which repository data to use')
    parser.add_argument('--samples', type=int, default=50,
                       help='Number of samples to test (default: 50)')
    parser.add_argument('--models', nargs='+', default=['gpt-4'],
                       choices=['gpt-4', 'gpt-4o', 'gpt-4-turbo', 'gpt-5',
                               'apertus-70b', 'apertus-8b', 'apertus-70b-bf16', 'apertus-8b-bf16'],
                       help='Models to test (default: gpt-4). Apertus options: apertus-70b, apertus-8b')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--wrapper', type=str, default='none',
                       choices=['none', 'dan', 'aim'],
                       help='Jailbreak wrapper to apply: none (default), dan (DAN 13.0), aim (AIM - MultiJail intentional)')
    parser.add_argument('--languages', nargs='+', default=None,
                       help='Languages to test for MultiJail (default: all). Options: en zh it vi ar ko th bn sw jv')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    print("="*80)
    print("MULTILINGUAL JAILBREAK ATTACK RUNNER (RAW COLLECTION)")
    print("="*80)
    print(f"Repository: {args.repo}")
    print(f"Samples: {args.samples}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Random seed: {args.seed}")
    if args.wrapper != 'none':
        print(f"Wrapper: {args.wrapper} ({'intentional scenario' if args.wrapper == 'aim' else 'DAN jailbreak'})")
    print("Note: Raw responses will be saved. Use src/eval/evaluate.py to score results.")
    print("="*80)

    # Load data based on repo selection
    if args.repo == 'csrt':
        data = load_csrt_data()
    elif args.repo == 'advbench':
        data = load_advbench_data()
    elif args.repo == 'cipherchat':
        data = load_cipherchat_data()
    elif args.repo == 'multijail':
        data = load_multijail_data(languages=args.languages)
    elif args.repo == 'arabizi':
        data = load_arabizi_data()
    elif args.repo == 'renellm':
        data = load_renellm_data()
    elif args.repo == 'safetybench':
        data = load_safetybench_data()
    elif args.repo == 'xsafety':
        data = load_xsafety_data()
    elif args.repo == 'all':
        print("\nLoading ALL repositories...")
        data = []
        data.extend(load_csrt_data())
        data.extend(load_advbench_data())
        data.extend(load_cipherchat_data())
        data.extend(load_multijail_data())
        data.extend(load_arabizi_data())
        data.extend(load_renellm_data())
        data.extend(load_safetybench_data())
        data.extend(load_xsafety_data())
        print(f"\n✓ Total combined: {len(data)} prompts")

    # Sample data
    if len(data) > args.samples:
        print(f"\nSampling {args.samples} from {len(data)} prompts...")
        data = random.sample(data, args.samples)

    # Run attacks (raw collection only - saves progressively after each model)
    saved_files = run_attacks(data, args.models, args.repo, args.wrapper, len(data))

    print(f"\n✓ Raw responses saved to {len(saved_files)} files (one per model)")
    print(f"\nTo evaluate results, run:")
    for filename in saved_files:
        print(f"  python src/eval/evaluate.py --input {filename} --method all")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
