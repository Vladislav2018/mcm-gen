"""Deduplicate benchmark files:
1. Remove duplicate formulas within each file (keep first occurrence)
2. Remove formulas from integrals/derivatives that appear in tasks (cross-file)
3. Remove formulas from integrals that appear in derivatives (cross-file)
"""
import json
import sys

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def dedup_within(data, label):
    """Remove duplicate formulas within a list, keeping first occurrence."""
    seen = set()
    result = []
    removed = 0
    for item in data:
        formula = item['ground_truth']['formula']
        if formula not in seen:
            seen.add(formula)
            result.append(item)
        else:
            removed += 1
            print(f"  [INTRA-DEDUP] {label}: removed duplicate '{formula}'")
    return result, removed

def dedup_cross(data, exclude_formulas, label, exclude_label):
    """Remove items from data whose formula appears in exclude_formulas set."""
    result = []
    removed = 0
    for item in data:
        formula = item['ground_truth']['formula']
        if formula not in exclude_formulas:
            result.append(item)
        else:
            removed += 1
            print(f"  [CROSS-DEDUP] {label}: removed '{formula}' (also in {exclude_label})")
    return result, removed

def main():
    tasks_file = "data/benchmark_tasks.jsonl"
    integrals_file = "data/benchmark_integrals.jsonl"
    derivatives_file = "data/benchmark_derivatives.jsonl"

    tasks = load_jsonl(tasks_file)
    integrals = load_jsonl(integrals_file)
    derivatives = load_jsonl(derivatives_file)

    print(f"BEFORE: Tasks={len(tasks)}, Integrals={len(integrals)}, Derivatives={len(derivatives)}")
    total_removed = 0

    # 1. Intra-file dedup
    print("\n--- Intra-file dedup ---")
    tasks, r = dedup_within(tasks, "tasks")
    total_removed += r
    integrals, r = dedup_within(integrals, "integrals")
    total_removed += r
    derivatives, r = dedup_within(derivatives, "derivatives")
    total_removed += r

    # 2. Cross-file dedup: remove from integrals & derivatives if formula is in tasks
    print("\n--- Cross-file dedup (vs tasks) ---")
    task_formulas = {t['ground_truth']['formula'] for t in tasks}
    integrals, r = dedup_cross(integrals, task_formulas, "integrals", "tasks")
    total_removed += r
    derivatives, r = dedup_cross(derivatives, task_formulas, "derivatives", "tasks")
    total_removed += r

    # 3. Cross-file dedup: remove from integrals if formula is in derivatives
    print("\n--- Cross-file dedup (integrals vs derivatives) ---")
    deriv_formulas = {d['ground_truth']['formula'] for d in derivatives}
    integrals, r = dedup_cross(integrals, deriv_formulas, "integrals", "derivatives")
    total_removed += r

    print(f"\nAFTER: Tasks={len(tasks)}, Integrals={len(integrals)}, Derivatives={len(derivatives)}")
    print(f"Total removed: {total_removed}")

    # Save
    save_jsonl(tasks_file, tasks)
    save_jsonl(integrals_file, integrals)
    save_jsonl(derivatives_file, derivatives)
    print("Files saved!")

if __name__ == "__main__":
    main()
