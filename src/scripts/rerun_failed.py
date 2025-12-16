#!/usr/bin/env python3
"""
Rerun failed (empty response) questions for a model.
Identifies questions with empty responses and reruns them.
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
import glob
import time

load_dotenv()

# Import model configs from run_experiment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import MODELS, get_client, query_model


def find_empty_responses(results_dir):
    """Find all responses with empty content."""
    empty_responses = []
    
    for filepath in glob.glob(f"{results_dir}/**/responses_*.jsonl", recursive=True):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    data = json.loads(line)
                    if not data.get('response'):
                        empty_responses.append({
                            'file': filepath,
                            'line_num': line_num,
                            'data': data
                        })
    
    return empty_responses


def rerun_single(args):
    """Rerun a single failed question."""
    item, client, model_id, provider, model_config, model_name, max_retries = args
    
    for attempt in range(max_retries):
        response_data = query_model(
            client, model_id, item['conversation'], provider, model_config=model_config
        )
        
        if response_data.get('success') and response_data.get('response'):
            return {
                **item,
                **response_data,
                'timestamp': datetime.now().isoformat(),
                'rerun_attempt': attempt + 1
            }
        
        if attempt < max_retries - 1:
            time.sleep(1)  # Brief pause before retry
    
    return {**item, **response_data, 'rerun_attempt': max_retries, 'still_empty': True}


def update_response_file(filepath, question_id, new_data):
    """Update a specific response in a file."""
    lines = []
    updated = False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data.get('question_id') == question_id:
                    lines.append(json.dumps(new_data, ensure_ascii=False) + '\n')
                    updated = True
                else:
                    lines.append(line)
    
    if updated:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    return updated


def main():
    parser = argparse.ArgumentParser(description="Rerun failed (empty response) questions")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--results-dir", type=str, default=None, help="Results directory")
    parser.add_argument("--workers", type=int, default=16, help="Parallel workers")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per question")
    parser.add_argument("--dry-run", action="store_true", help="Just show what would be rerun")
    
    args = parser.parse_args()
    
    model_config = MODELS[args.model]
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = args.results_dir or os.path.join(base_dir, f"results/{args.model}")
    
    print(f"Scanning {results_dir} for empty responses...")
    empty_responses = find_empty_responses(results_dir)
    
    print(f"Found {len(empty_responses)} empty responses")
    
    if args.dry_run:
        for item in empty_responses:
            print(f"  {item['file']}: {item['data']['question_id']}")
        return
    
    if not empty_responses:
        print("No empty responses to rerun!")
        return
    
    # Get client
    client = get_client(model_config)
    provider = model_config.get('provider', 'openai')
    model_id = model_config['model_id']
    
    # Prepare tasks
    task_args = [
        (item['data'], client, model_id, provider, model_config, args.model, args.max_retries)
        for item in empty_responses
    ]
    
    # Run in parallel
    results = []
    file_lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(rerun_single, task): empty_responses[i] 
                   for i, task in enumerate(task_args)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Rerunning"):
            original = futures[future]
            result = future.result()
            
            if result.get('response'):
                # Update the file
                with file_lock:
                    update_response_file(original['file'], result['question_id'], result)
                tqdm.write(f"✓ Fixed: {result['question_id']}")
            else:
                tqdm.write(f"✗ Still empty: {result['question_id']}")
            
            results.append(result)
    
    # Summary
    fixed = sum(1 for r in results if r.get('response'))
    still_empty = len(results) - fixed
    print(f"\n{'='*60}")
    print(f"Fixed: {fixed}/{len(results)}")
    print(f"Still empty: {still_empty}")


if __name__ == "__main__":
    main()
