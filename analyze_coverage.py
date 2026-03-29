"""Analyze coverage: which task formulas have integrals/derivatives, and find duplicates."""
import json

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

tasks = load_jsonl('benchmark_tasks.jsonl')
integrals = load_jsonl('benchmark_integrals.jsonl')
derivs = load_jsonl('benchmark_derivatives.jsonl')

print(f"Tasks: {len(tasks)}")
print(f"Integrals: {len(integrals)}")
print(f"Derivatives: {len(derivs)}")

# Extract formulas from tasks
task_formulas = set()
for t in tasks:
    task_formulas.add(t['ground_truth']['formula'])
print(f"\nUnique task formulas: {len(task_formulas)}")

# Extract source formulas from integrals/derivatives
integral_sources = set()
for i in integrals:
    src = i.get('source', '')
    if src.startswith('integral_of_'):
        integral_sources.add(src[len('integral_of_'):])

deriv_sources = set()
for d in derivs:
    src = d.get('source', '')
    if src.startswith('derivative_of_'):
        deriv_sources.add(src[len('derivative_of_'):])

# Find missing
missing_integral = task_formulas - integral_sources
missing_deriv = task_formulas - deriv_sources

print(f"\n=== MISSING INTEGRALS ({len(missing_integral)}) ===")
for f in sorted(missing_integral):
    print(f"  {f}")

print(f"\n=== MISSING DERIVATIVES ({len(missing_deriv)}) ===")
for f in sorted(missing_deriv):
    print(f"  {f}")

# Check for duplicates WITHIN each file
print(f"\n=== DUPLICATE CHECK ===")

# Within benchmark_tasks.jsonl - by formula
task_formula_count = {}
for t in tasks:
    f = t['ground_truth']['formula']
    task_formula_count[f] = task_formula_count.get(f, 0) + 1
dup_tasks = {f: c for f, c in task_formula_count.items() if c > 1}
if dup_tasks:
    print(f"\nDuplicate formulas in benchmark_tasks.jsonl ({len(dup_tasks)}):")
    for f, c in dup_tasks.items():
        print(f"  '{f}' appears {c} times")
else:
    print("\nNo duplicate formulas in benchmark_tasks.jsonl")

# Within benchmark_integrals.jsonl - by formula
integ_formula_count = {}
for i in integrals:
    f = i['ground_truth']['formula']
    integ_formula_count[f] = integ_formula_count.get(f, 0) + 1
dup_integs = {f: c for f, c in integ_formula_count.items() if c > 1}
if dup_integs:
    print(f"\nDuplicate formulas in benchmark_integrals.jsonl ({len(dup_integs)}):")
    for f, c in dup_integs.items():
        print(f"  '{f}' appears {c} times")
else:
    print("\nNo duplicate formulas in benchmark_integrals.jsonl")

# Within benchmark_derivatives.jsonl - by formula
deriv_formula_count = {}
for d in derivs:
    f = d['ground_truth']['formula']
    deriv_formula_count[f] = deriv_formula_count.get(f, 0) + 1
dup_derivs = {f: c for f, c in deriv_formula_count.items() if c > 1}
if dup_derivs:
    print(f"\nDuplicate formulas in benchmark_derivatives.jsonl ({len(dup_derivs)}):")
    for f, c in dup_derivs.items():
        print(f"  '{f}' appears {c} times")
else:
    print("\nNo duplicate formulas in benchmark_derivatives.jsonl")

# Cross-file duplicates (formulas appearing in multiple files)
print(f"\n=== CROSS-FILE DUPLICATE CHECK ===")
task_set = set(t['ground_truth']['formula'] for t in tasks)
integ_set = set(i['ground_truth']['formula'] for i in integrals)
deriv_set = set(d['ground_truth']['formula'] for d in derivs)

ti = task_set & integ_set
td = task_set & deriv_set
id_ = integ_set & deriv_set

if ti:
    print(f"\nFormulas in BOTH tasks AND integrals ({len(ti)}):")
    for f in sorted(ti):
        print(f"  {f}")
if td:
    print(f"\nFormulas in BOTH tasks AND derivatives ({len(td)}):")
    for f in sorted(td):
        print(f"  {f}")
if id_:
    print(f"\nFormulas in BOTH integrals AND derivatives ({len(id_)}):")
    for f in sorted(id_):
        print(f"  {f}")
if not ti and not td and not id_:
    print("\nNo cross-file duplicate formulas found.")
