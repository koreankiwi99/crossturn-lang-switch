#!/usr/bin/env python3
"""Analyze MultiChallenge dataset composition"""
import json
from collections import Counter, defaultdict

data_path = "data/multi-challenge/data/benchmark_questions.jsonl"

# Load data
with open(data_path, 'r') as f:
    data = [json.loads(line) for line in f if line.strip()]

print("=" * 70)
print("MULTICHALLENGE DATASET COMPOSITION")
print("=" * 70)

# Total count
print(f"\nTotal questions: {len(data)}")

# By axis
print("\n--- By Axis ---")
axis_counts = Counter(d['AXIS'] for d in data)
for axis, count in sorted(axis_counts.items(), key=lambda x: -x[1]):
    pct = count / len(data) * 100
    print(f"  {axis}: {count} ({pct:.1f}%)")

# Conversation length stats
print("\n--- Conversation Length (turns) ---")
conv_lengths = [len(d['CONVERSATION']) for d in data]
print(f"  Min turns: {min(conv_lengths)}")
print(f"  Max turns: {max(conv_lengths)}")
print(f"  Avg turns: {sum(conv_lengths)/len(conv_lengths):.1f}")

# Distribution by turn count
turn_dist = Counter(conv_lengths)
print("\n  Overall Distribution:")
for turns, count in sorted(turn_dist.items()):
    print(f"    {turns} turns: {count} questions")

# === TURN DISTRIBUTION BY AXIS ===
print("\n" + "=" * 70)
print("TURN DISTRIBUTION BY AXIS")
print("=" * 70)

axis_turn_dist = defaultdict(lambda: defaultdict(int))
axis_turn_stats = {}

for d in data:
    axis = d['AXIS']
    turns = len(d['CONVERSATION'])
    axis_turn_dist[axis][turns] += 1

for axis in sorted(axis_counts.keys()):
    turns_data = axis_turn_dist[axis]
    all_turns = []
    for t, c in turns_data.items():
        all_turns.extend([t] * c)

    min_t = min(all_turns)
    max_t = max(all_turns)
    avg_t = sum(all_turns) / len(all_turns)

    print(f"\n{axis} ({axis_counts[axis]} questions)")
    print(f"  Range: {min_t}-{max_t} turns, Avg: {avg_t:.1f}")
    print(f"  Distribution:")
    for turns, count in sorted(turns_data.items()):
        bar = "█" * count
        print(f"    {turns:2d} turns: {count:3d} {bar}")

# Pass criteria
print("\n" + "=" * 70)
print("--- Pass Criteria ---")
criteria_counts = Counter(d['PASS_CRITERIA'] for d in data)
for criteria, count in criteria_counts.items():
    print(f"  {criteria}: {count}")

print("\n" + "=" * 70)
