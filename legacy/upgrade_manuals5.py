import json
import numpy as np
import sympy as sp
import uuid
import itertools
import random
from src.config import ComplexityConfig
from src.sampler import TaskExporter
from upgrade_manuals4 import get_points, calculate_metadata_safe, adaptive_sample, dumps_compact

MANUAL_FILE = "manual_formulas.json"

def get_rich_formulas():
    formulas = {
        "0,0,0": ["x", "2*x", "x+1"],
        "1,0,0": ["x**2", "x**2 - x", "2*x**2 + 1", "0.5*x**2"],
        "2,0,0": ["(x+1)**2", "(x-0.5)**2", "(2*x+1)**2"],
        "3,0,0": ["((x+1)**2 + 1)**2", "(x**2 - 0.5)**3"],
        
        "0,1,0": ["sin(x)", "cos(x)", "exp(x)", "2**x", "3**x", "0.5**x"],
        "1,1,0": ["sin(x) + cos(x)", "x*sin(x)", "x**2 * cos(x)", "exp(x) - x", "2**x + x**2"],
        "2,1,0": ["sin(x**2)", "cos(x+1)", "exp(-x**2)", "2**(x-1)", "sin(cos(x))"],
        "3,1,0": ["sin(exp(x))", "exp(sin(x))", "2**(sin(x))", "sin(x**2 + cos(x))"],
        
        "0,2,0": ["Abs(x)", "sign(x)"],
        "1,2,0": ["Abs(x) + x", "x*sign(x)", "Abs(x)*sin(x)", "Abs(sin(x))"],
        "2,2,0": ["Abs(x**2 - 1)", "sign(sin(x))", "Abs(x+0.5)"],
        "3,2,0": ["Abs(sin(x**2))", "sign(x**2 - 0.5)"],
        
        "0,3,0": ["gamma(Abs(x) + 1.1)", "besselj(0, Abs(x))", "erf(x)"],
        "1,3,0": ["gamma(Abs(x)+1.1) + x", "x*besselj(0, Abs(x))", "sin(x)*erf(x)"],
        "2,3,0": ["gamma(Abs(x**2)+1.1)", "besselj(1, Abs(x))", "erf(x**2)"],
        "3,3,0": ["gamma(Abs(sin(x))+1.1)", "besselj(0, Abs(x**2+1))", "erf(sin(x))"]
    }
    
    # Generate C=1 (Periodic/Symmetric) by wrapping
    # Generate C=2 (Asymptotes) by dividing by (x - offset)
    # Generate C=3 (Singularities/High freq) by injecting x**2 in denom or high freq args
    
    offsets = [0.1, 0.2, -0.15, -0.3, 1.1, -1.2]
    
    full_matrix = {}
    
    for a in range(4):
        for b in range(4):
            k0 = f"{a},{b},0"
            if k0 in formulas:
                full_matrix[k0] = formulas[k0]
                
                # C=1 (Periodic variants)
                k1 = f"{a},{b},1"
                full_matrix[k1] = []
                for f in formulas[k0]:
                    full_matrix[k1].append(f"sin({f})")
                    full_matrix[k1].append(f"cos({f})")
                
                # C=2 (Asymptotes)
                k2 = f"{a},{b},2"
                full_matrix[k2] = []
                for i, f in enumerate(formulas[k0]):
                    off = offsets[i % len(offsets)]
                    full_matrix[k2].append(f"({f}) / (x - {off})")
                    
                # C=3 (Chaos/Singularities)
                k3 = f"{a},{b},3"
                full_matrix[k3] = []
                for i, f in enumerate(formulas[k0]):
                    off = offsets[(i+1) % len(offsets)]
                    full_matrix[k3].append(f"({f}) * sin(1/(x - {off}))")
                    # Also factorial-like behavior for high complexity
                    if b >= 1 and a >= 1:
                        full_matrix[k3].append(f"({f}) * gamma(Abs(x)+1.1)")
                        
    return full_matrix

def inject_rich_formulas():
    with open(MANUAL_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    rich_dict = get_rich_formulas()
    
    new_tasks_added = 0
    
    for k, exprs in rich_dict.items():
        if k not in data:
            data[k] = []
            
        # Get existing formulas to avoid exact duplicates
        existing_formula_strs = [t["ground_truth"]["formula"].replace(" ", "") for t in data[k]]
        
        for expr_str in exprs:
            clean_str = expr_str.replace(" ", "")
            if clean_str in existing_formula_strs:
                continue
                
            x_sym = sp.Symbol('x', real=True)
            try:
                expr = sp.parse_expr(expr_str, local_dict={'x': x_sym})
            except Exception:
                continue
                
            pts = adaptive_sample(expr_str, n=25)
            meta_success, metadata, meta_err = calculate_metadata_safe(expr, timeout=3)
            
            if not meta_success:
                continue
                
            pts_success, x_vals, y_vals = get_points(expr, custom_x=pts, n=25)
            
            if not pts_success:
                pts_success, x_vals, y_vals = get_points(expr, custom_x=np.linspace(-3, 3, 25), n=25)
                
            if pts_success:
                task_id = str(uuid.uuid4())[:8]
                points_data = [{"x": round(float(xi), 3), "y": round(float(yi), 4)} for xi, yi in zip(x_vals, y_vals)]
                
                a, b, c = map(int, k.split(','))
                task = {
                    "task_id": f"MCM_{a}{b}{c}_{task_id}",
                    "complexity_vector": {"a": a, "b": b, "c": c},
                    "prompt_data": {"points": points_data},
                    "ground_truth": {
                        "formula": str(expr),
                        "latex": sp.latex(expr),
                        "properties": metadata
                    },
                    "source": "manual_rich"
                }
                data[k].append(task)
                existing_formula_strs.append(clean_str)
                new_tasks_added += 1
                print(f"Added rich {k}: {expr_str}")
                
    if new_tasks_added > 0:
        with open(MANUAL_FILE, 'w', encoding='utf-8') as f:
            f.write(dumps_compact(data))
        print(f"Successfully injected {new_tasks_added} new rich formulas!")
    else:
        print("No new formulas were successfully generated.")

if __name__ == "__main__":
    inject_rich_formulas()
