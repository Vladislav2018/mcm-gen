"""MCM-Gen Quality Audit

Analyzes the benchmark dataset for:
1. Compliance: do functions match their claimed complexity vector?
2. Diversity: how different are functions within each class?
3. Coverage: distribution across the 64 classes

Usage: python quality_audit.py
"""

import json
import sympy as sp
import numpy as np
from collections import Counter, defaultdict

OUTPUT_FILE = "data/benchmark_tasks.jsonl"


def load_tasks(filepath):
    """Load all tasks from JSONL file."""
    tasks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


def check_b_compliance(formula_str, target_b):
    """Check if formula uses operators from the correct B level."""
    b_markers = {
        0: [],  # B=0 has no specific markers
        1: ['sin', 'cos', 'tan', 'exp', 'log'],
        2: ['Abs', 'floor', 'sign', 'Piecewise', 'factorial'],
        3: ['gamma', 'besselj', 'erf', 'zeta'],
    }

    if target_b == 0:
        # Check that NO higher-level operators are used
        for lvl in [1, 2, 3]:
            for marker in b_markers[lvl]:
                if marker in formula_str:
                    return False, f"Uses {marker} (B={lvl}) but target is B=0"
        return True, "OK"

    # For B >= 1: must contain at least one operator from target level
    markers = b_markers.get(target_b, [])
    found = [m for m in markers if m in formula_str]
    if not found:
        return False, f"No B={target_b} operators found"
    return True, f"Found: {', '.join(found)}"


def check_c_compliance(formula_str, properties, target_c):
    """Check if function behavior matches target C level."""
    sings = properties.get('singularities', [])
    is_periodic = properties.get('is_periodic', False)
    domain = properties.get('domain', 'Reals')

    if target_c == 0:
        # Should have no singularities and not be periodic
        has_sings = bool(sings) and sings != ['analysis_timeout'] and sings != ['analysis_error']
        if has_sings:
            return False, f"Has singularities: {sings[:3]}..."
        if is_periodic:
            return False, "Is periodic but target is C=0"
        return True, "OK"

    if target_c == 1:
        if is_periodic:
            return True, "Periodic detected"
        # Check string for trig patterns
        if any(t in formula_str for t in ['sin(', 'cos(']):
            return True, "Has trig (likely periodic)"
        return False, "No periodicity detected"

    if target_c == 2:
        if sings and sings != ['analysis_timeout'] and sings != ['analysis_error']:
            return True, f"Singularities: {sings[:3]}"
        if '/' in formula_str and '(x' in formula_str:
            return True, "Has rational form (likely singular)"
        return False, "No singularities detected"

    if target_c == 3:
        # C3 is broader: multiple sings, oscillating sings, chaos
        if 'sin(1/' in formula_str or 'cos(1/' in formula_str:
            return True, "Oscillating singularity"
        if sings and len(sings) >= 2:
            return True, f"Multiple singularities: {len(sings)}"
        if '/' in formula_str:
            return True, "Has rational form (complex behavior)"
        return False, "No complex behavior detected"

    return True, "Unknown C level"


def compute_diversity(formulas):
    """Compute simple diversity metrics for a set of formula strings."""
    if len(formulas) <= 1:
        return {"count": len(formulas), "unique_ops": 0, "avg_length": 0}

    ops_sets = []
    lengths = []
    for f in formulas:
        ops = set()
        for op in ['sin', 'cos', 'exp', 'log', 'Abs', 'floor', 'gamma', 'erf',
                    'besselj', 'sign', 'Piecewise', '**', '/', '+', '-']:
            if op in f:
                ops.add(op)
        ops_sets.append(ops)
        lengths.append(len(f))

    # Count unique operator combinations
    unique_combos = len(set(frozenset(s) for s in ops_sets))

    return {
        "count": len(formulas),
        "unique_op_combos": unique_combos,
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }


def run_audit():
    """Run full quality audit."""
    tasks = load_tasks(OUTPUT_FILE)
    print(f"=== MCM-Gen QUALITY AUDIT ===")
    print(f"Total tasks: {len(tasks)}\n")

    # === 1. Coverage ===
    class_counts = Counter()
    source_counts = Counter()
    for t in tasks:
        cv = t['complexity_vector']
        class_counts[(cv['a'], cv['b'], cv['c'])] += 1
        source_counts[t.get('source', 'unknown')] += 1

    print("--- Coverage ---")
    print(f"Classes filled: {len(class_counts)}/64")
    print(f"Min tasks/class: {min(class_counts.values())}")
    print(f"Max tasks/class: {max(class_counts.values())}")
    print(f"Avg tasks/class: {sum(class_counts.values()) / len(class_counts):.1f}")
    under_10 = sum(1 for v in class_counts.values() if v < 10)
    under_15 = sum(1 for v in class_counts.values() if v < 15)
    print(f"Classes with < 10 tasks: {under_10}")
    print(f"Classes with < 15 tasks: {under_15}")
    print(f"\nSources: {dict(source_counts)}")

    # === 2. B-Compliance ===
    print("\n--- B-Axis Compliance ---")
    b_results = defaultdict(lambda: {"pass": 0, "fail": 0, "issues": []})
    for t in tasks:
        b = t['complexity_vector']['b']
        formula = t['ground_truth']['formula']
        ok, msg = check_b_compliance(formula, b)
        if ok:
            b_results[b]["pass"] += 1
        else:
            b_results[b]["fail"] += 1
            if len(b_results[b]["issues"]) < 3:
                b_results[b]["issues"].append(f"{formula[:40]}... -> {msg}")

    for b in range(4):
        r = b_results[b]
        total = r["pass"] + r["fail"]
        pct = r["pass"] / total * 100 if total > 0 else 0
        print(f"  B={b}: {r['pass']}/{total} pass ({pct:.0f}%)")
        for issue in r["issues"]:
            print(f"    ! {issue}")

    # === 3. C-Compliance ===
    print("\n--- C-Axis Compliance ---")
    c_results = defaultdict(lambda: {"pass": 0, "fail": 0, "issues": []})
    for t in tasks:
        c = t['complexity_vector']['c']
        formula = t['ground_truth']['formula']
        props = t['ground_truth'].get('properties', {})
        ok, msg = check_c_compliance(formula, props, c)
        if ok:
            c_results[c]["pass"] += 1
        else:
            c_results[c]["fail"] += 1
            if len(c_results[c]["issues"]) < 3:
                c_results[c]["issues"].append(f"{formula[:40]}... -> {msg}")

    for c in range(4):
        r = c_results[c]
        total = r["pass"] + r["fail"]
        pct = r["pass"] / total * 100 if total > 0 else 0
        print(f"  C={c}: {r['pass']}/{total} pass ({pct:.0f}%)")
        for issue in r["issues"]:
            print(f"    ! {issue}")

    # === 4. Diversity ===
    print("\n--- Diversity per class ---")
    class_formulas = defaultdict(list)
    for t in tasks:
        cv = t['complexity_vector']
        class_formulas[(cv['a'], cv['b'], cv['c'])].append(t['ground_truth']['formula'])

    low_diversity = []
    for key in sorted(class_formulas.keys()):
        div = compute_diversity(class_formulas[key])
        if div['count'] >= 3 and div['unique_op_combos'] <= 1:
            low_diversity.append((key, div))

    if low_diversity:
        print(f"  Classes with low operator diversity ({len(low_diversity)}):")
        for key, div in low_diversity[:10]:
            print(f"    ({key[0]},{key[1]},{key[2]}): "
                  f"{div['count']} tasks, {div['unique_op_combos']} unique op combos")
    else:
        print("  All classes have good operator diversity [OK]")

    # === 5. Summary ===
    total_b_pass = sum(b_results[b]["pass"] for b in range(4))
    total_c_pass = sum(c_results[c]["pass"] for c in range(4))
    print(f"\n--- SUMMARY ---")
    print(f"B-compliance: {total_b_pass}/{len(tasks)} ({total_b_pass/len(tasks)*100:.0f}%)")
    print(f"C-compliance: {total_c_pass}/{len(tasks)} ({total_c_pass/len(tasks)*100:.0f}%)")
    print(f"Coverage: {len(class_counts)}/64 classes")


if __name__ == "__main__":
    run_audit()
