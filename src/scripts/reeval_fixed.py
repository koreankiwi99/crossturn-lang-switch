#!/usr/bin/env python3
"""Re-evaluate fixed responses that had verdict=None."""

import os
import json
import glob
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Setup
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
JUDGE_MODEL = "gpt-4o-2024-08-06"

# The 5 fixed question IDs
fixed_ids = [
    '6745673294ace5b84220c471_zh_reverse',
    '6765f839a02fd129919ef243_zh_reverse',
    '6781aad8e76fccc700cb9025_ar_reverse',
    '6765f1c843f5c6a2c6f27ea6_ar_reverse',
    '6781a948112db930b29e011e_zh'
]


def load_judge_prompt(base_dir):
    """Load judge prompt from prompts/judge_prompt.txt (same as evaluate.py)."""
    prompt_path = os.path.join(base_dir, "prompts/judge_prompt.txt")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    judge_prompt_template = load_judge_prompt(base_dir)

    # Find and re-evaluate
    for responses_file in glob.glob(f'{base_dir}/results/gemini-3-pro/**/responses_*.jsonl', recursive=True):
        # Find corresponding evaluated files (process all of them)
        eval_dir = os.path.dirname(responses_file)
        eval_files = glob.glob(f'{eval_dir}/evaluated_*.jsonl')
        if not eval_files:
            continue

        # Load responses
        responses = {}
        with open(responses_file) as f:
            for line in f:
                d = json.loads(line)
                if d['question_id'] in fixed_ids and d.get('response'):
                    responses[d['question_id']] = d

        if not responses:
            continue

        # Process each evaluated file in this directory
        for eval_file in eval_files:
            # Load and update evaluations
            eval_data = []
            updated = False
            with open(eval_file) as f:
                for line in f:
                    d = json.loads(line)
                    # Check if this needs re-evaluation:
                    # - question_id is in fixed list
                    # - evaluated file has empty response but we now have actual response
                    if d['question_id'] in responses and not d.get('response'):
                        # Re-evaluate
                        resp = responses[d['question_id']]
                        prompt = judge_prompt_template.format(
                            response=resp['response'],
                            target_question=resp['target_question']
                        )

                        try:
                            result = client.chat.completions.create(
                                model=JUDGE_MODEL,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0,
                                max_tokens=1024
                            )
                            judge_text = result.choices[0].message.content

                            # Parse verdict (same logic as evaluate.py)
                            if judge_text is None or judge_text.strip() == "":
                                verdict = "ERROR"
                            else:
                                judge_text_upper = judge_text.upper().strip()
                                if judge_text_upper.endswith("YES"):
                                    verdict = "YES"
                                elif judge_text_upper.endswith("NO"):
                                    verdict = "NO"
                                else:
                                    last_line = judge_text.strip().split('\n')[-1].upper()
                                    verdict = "YES" if "YES" in last_line else "NO"

                            pass_criteria = d.get("pass_criteria", "YES")
                            passed = (verdict == pass_criteria)

                            # Update evaluation in same structure as evaluate.py
                            d['evaluation'] = {
                                "judge_model": JUDGE_MODEL,
                                "verdict": verdict,
                                "pass_criteria": pass_criteria,
                                "passed": passed,
                                "reasoning": judge_text,
                            }
                            # Also update the response field from the fixed data
                            d['response'] = resp['response']

                            print(f"  Re-evaluated {d['question_id']}: {verdict} (passed={passed})")
                            updated = True
                        except Exception as e:
                            print(f"  Error re-evaluating {d['question_id']}: {e}")

                    eval_data.append(d)

            if updated:
                with open(eval_file, 'w') as f:
                    for d in eval_data:
                        f.write(json.dumps(d, ensure_ascii=False) + '\n')
                print(f"Updated: {eval_file}")

    print("\nDone!")

if __name__ == "__main__":
    main()
