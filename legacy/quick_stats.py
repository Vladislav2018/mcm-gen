"""Quick stats on benchmark distribution - no SymPy needed."""
import json
from collections import Counter

counts = Counter()
sources = Counter()
total = 0

with open('benchmark_tasks.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        cv = d['complexity_vector']
        key = (cv['a'], cv['b'], cv['c'])
        counts[key] += 1
        sources[d.get('source', 'unknown')] += 1
        total += 1

print(f"Total tasks: {total}")
print(f"\nSources: {dict(sources)}")
print(f"\nDistribution across {len(counts)} classes:")
for k in sorted(counts.keys()):
    print(f"  ({k[0]},{k[1]},{k[2]}): {counts[k]}")

# Check coverage
print(f"\nCoverage: {len(counts)}/64 classes filled ({len(counts)/64*100:.0f}%)")
empty = [(a,b,c) for a in range(4) for b in range(4) for c in range(4) if (a,b,c) not in counts]
if empty:
    print(f"Empty classes ({len(empty)}):")
    for e in empty:
        print(f"  ({e[0]},{e[1]},{e[2]})")
