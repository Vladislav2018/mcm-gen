import json
import numpy as np
import sympy as sp

def get_ast_depth(expr):
    if not expr.args:
        return 0
    return 1 + max(get_ast_depth(arg) for arg in expr.args)

def analyze_dataset(filename):
    manual = []
    auto = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            src = data.get("source", "old")
            if src == "manual":
                manual.append(data)
            elif src == "auto":
                auto.append(data)
                
    def get_stats(dataset, name):
        if not dataset:
            print(f"--- No data for {name} ---")
            return
            
        print(f"\n=== M: {name} (N = {len(dataset)}) ===")
        
        classes = {}
        depths = []
        for d in dataset:
            cv = d["complexity_vector"]
            k = f"{cv['a']},{cv['b']},{cv['c']}"
            classes[k] = classes.get(k, 0) + 1
            
            formula = d["ground_truth"]["formula"]
            try:
                e = sp.parse_expr(formula)
                depths.append(get_ast_depth(e))
            except:
                pass
            
        print(f"Coverage of ABC classes: {len(classes)} / 64")
        if depths:
            print(f"Avg AST depth: {np.mean(depths):.2f} (Max: {max(depths)})")
        
        y_ranges = []
        periodic_count = 0
        sings_count = 0
        
        for d in dataset:
            pts = d.get("prompt_data", {}).get("points", [])
            if pts:
                ys = [p["y"] for p in pts]
                y_ranges.append(max(ys) - min(ys))
                
            props = d.get("ground_truth", {}).get("properties", {})
            if props.get("is_periodic"):
                periodic_count += 1
            if len(props.get("singularities", [])) > 0:
                sings_count += 1
                
        if y_ranges:
            print(f"Avg Y-Range: {np.mean(y_ranges):.1f} (Median: {np.median(y_ranges):.1f})")
            print(f"Max Y-Range: {max(y_ranges):.1f}")
        print(f"Periodic functions: {periodic_count} ({periodic_count/len(dataset)*100:.1f}%)")
        print(f"Functions w/ singularities: {sings_count} ({sings_count/len(dataset)*100:.1f}%)")

    print(f"Total new tagged entries: {len(manual) + len(auto)}")
    get_stats(manual, "Manual Generations (Curated & Smart Sampled)")
    get_stats(auto, "Auto Generations (MCM-Gen Pipeline)")

if __name__ == "__main__":
    analyze_dataset("benchmark_tasks.jsonl")
