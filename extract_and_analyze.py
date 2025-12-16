#!/usr/bin/env python3
"""Extract INFERENCE_MEMORY and INSTRUCTION_RETENTION, sorted by turn count"""
import json
import os
from collections import Counter

# Load data
with open("data/multi-challenge/data/benchmark_questions.jsonl", 'r') as f:
    all_data = [json.loads(line) for line in f if line.strip()]

# Filter to target axes
target_axes = ["INFERENCE_MEMORY", "INSTRUCTION_RETENTION"]
filtered = [d for d in all_data if d['AXIS'] in target_axes]

# Add turn count
for item in filtered:
    item['turn_count'] = len(item['CONVERSATION'])

# Sort by turn count
filtered.sort(key=lambda x: (x['AXIS'], x['turn_count']))

print("=" * 70)
print("EXTRACTED DATASET: INFERENCE_MEMORY + INSTRUCTION_RETENTION")
print("=" * 70)
print(f"\nTotal extracted: {len(filtered)} questions")

# Analyze by axis
for axis in target_axes:
    axis_data = [d for d in filtered if d['AXIS'] == axis]
    turn_counts = [d['turn_count'] for d in axis_data]
    turn_dist = Counter(turn_counts)

    print(f"\n{'='*50}")
    print(f"{axis} ({len(axis_data)} questions)")
    print(f"{'='*50}")
    print(f"{'Turns':<10} {'Count':<10} {'Visual'}")
    print("-" * 50)

    for turns in sorted(turn_dist.keys()):
        count = turn_dist[turns]
        bar = "█" * count
        print(f"{turns:<10} {count:<10} {bar}")

    print("-" * 50)
    print(f"Range: {min(turn_counts)}-{max(turn_counts)} turns")
    print(f"Average: {sum(turn_counts)/len(turn_counts):.1f} turns")

# Save extracted data
os.makedirs("data/extracted", exist_ok=True)

# Combined file
with open("data/extracted/multichallenge_INFERENCE_INSTRUCTION.jsonl", 'w') as f:
    for item in filtered:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Individual files
for axis in target_axes:
    axis_data = [d for d in filtered if d['AXIS'] == axis]
    with open(f"data/extracted/multichallenge_{axis}.jsonl", 'w') as f:
        for item in axis_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n{'='*70}")
print("FILES SAVED:")
print("  data/extracted/multichallenge_INFERENCE_INSTRUCTION.jsonl (combined)")
print("  data/extracted/multichallenge_INFERENCE_MEMORY.jsonl")
print("  data/extracted/multichallenge_INSTRUCTION_RETENTION.jsonl")
print("=" * 70)
