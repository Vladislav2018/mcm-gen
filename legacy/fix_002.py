import json
import numpy as np
import sympy as sp
import uuid
from src.config import ComplexityConfig
from src.sampler import TaskExporter
from upgrade_manuals4 import get_points, calculate_metadata_safe, adaptive_sample, dumps_compact

MANUAL_FILE = "manual_formulas.json"

def fix_002():
    with open(MANUAL_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    k = "0,0,2"
    formulas = ["1/(x-0.1)", "1/(x+0.2)", "x/(x-1.1)"]
    config = ComplexityConfig(0, 0, 2)
    
    if k not in data:
        data[k] = []
        
    for expr_str in formulas:
        x_sym = sp.Symbol('x', real=True)
        expr = sp.parse_expr(expr_str, local_dict={'x': x_sym})
        
        pts = adaptive_sample(expr_str, n=25)
        meta_success, metadata, meta_err = calculate_metadata_safe(expr, timeout=5)
        pts_success, x_vals, y_vals = get_points(expr, custom_x=pts, n=25)
        
        if not pts_success:
            pts_success, x_vals, y_vals = get_points(expr, custom_x=np.linspace(-3, 3, 25), n=25)
            
        if pts_success:
            task_id = str(uuid.uuid4())[:8]
            points_data = [{"x": round(float(xi), 3), "y": round(float(yi), 4)} for xi, yi in zip(x_vals, y_vals)]
            
            task = {
                "task_id": f"MCM_002_{task_id}",
                "complexity_vector": {"a": 0, "b": 0, "c": 2},
                "prompt_data": {"points": points_data},
                "ground_truth": {
                    "formula": str(expr),
                    "latex": sp.latex(expr),
                    "properties": metadata
                },
                "source": "manual"
            }
            data[k].append(task)
            print(f"Added {k}: {expr_str}")
            
    with open(MANUAL_FILE, 'w', encoding='utf-8') as f:
        f.write(dumps_compact(data))

if __name__ == "__main__":
    fix_002()
