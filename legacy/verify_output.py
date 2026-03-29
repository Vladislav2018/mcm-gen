import json

def analyze_output(filename):
    classes = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            cv = data['complexity_vector']
            key = f"{cv['a']},{cv['b']},{cv['c']}"
            if key not in classes:
                classes[key] = []
            classes[key].append(data['ground_truth']['formula'])
            
    print(f"Total equations: {sum(len(v) for v in classes.values())}")
    for k, v in sorted(classes.items()):
        print(f"Class {k}: {len(v)} equations")
        if k in ['1,1,1', '2,1,2', '3,3,3', '0,0,0']:
            for eq in v[:2]:
                print(f"  - {eq}")

analyze_output("benchmark_tasks.jsonl")
