import json
import numpy as np
import sympy as sp
import os
import multiprocessing
from src.config import ComplexityConfig
from src.sampler import DatasetSampler, TaskExporter

MANUAL_FILE = "manual_formulas.json"

extra_formulas = {
    "0,0,0": ["10*x - 3", "-0.5*x + 1", "x", "3"],
    "0,1,0": ["sin(x) + cos(x)", "exp(-x)", "2*cos(x)"],
    "0,2,0": ["abs(x - 2)", "abs(x + 1) - x"],
    "1,0,0": ["x**2", "-x**2 + 4*x - 4", "0.1*x**3"],
    "1,1,0": ["sin(x)*exp(-0.1*x)", "cos(x)**2"],
    "1,2,0": ["abs(x**2 - 4)", "x * abs(x)"],
    "2,0,0": ["x**3 - x", "x**4 - 2*x**2 + 1"],
    "2,1,0": ["sin(x**2)", "exp(sin(x))"],
    "2,2,0": ["abs(x**3)", "Piecewise((x**2, x>0), (-x**2, True))"],
    "0,0,2": ["1/x", "1/(x+1)", "x/(x-2)"],
    "1,0,2": ["1/(x**2 - 1)", "x**2/(x - 3)"],
    "2,0,2": ["(x**3 + 1)/(x**2 - 4)"],
    "3,3,3": ["gamma(x)*sin(1/x)", "zeta(x)*log(abs(x))"]
}

def adaptive_sample(expr_str, n=25):
    """
    Evaluates function on a dense grid and samples based on arc length 
    to capture "interesting" features perfectly.
    """
    x_sym = sp.Symbol('x', real=True)
    try:
        expr = sp.parse_expr(expr_str)
        safe_modules = [
            {'factorial': lambda n: np.clip(np.array(n, dtype=float), 0, 12)}, 
            "scipy", 
            "numpy"
        ]
        f = sp.lambdify(x_sym, expr, modules=safe_modules)
    except:
        return list(np.round(np.linspace(-3, 3, n), 3))

    dense_x = np.linspace(-6, 6, 1000)
    with np.errstate(all='ignore'):
        try:
            y = f(dense_x)
        except Exception:
            y = np.array([f(val) for val in dense_x])
            
        if np.isscalar(y): y = np.full_like(dense_x, y)
        y = np.array(y, dtype=float)

    # Filter invalid points
    valid_mask = np.isfinite(y) & (np.abs(y) < 1000)
    if not np.any(valid_mask):
        return list(np.round(np.linspace(-3, 3, n), 3))

    valid_x = dense_x[valid_mask]
    valid_y = y[valid_mask]

    if len(valid_x) < n:
        return [round(float(v), 3) for v in valid_x]

    # Calculate arc length
    dx = np.diff(valid_x)
    dy = np.diff(valid_y)
    
    # Scale x and y so they contribute equally to 'length'
    range_x = np.max(valid_x) - np.min(valid_x)
    range_y = np.max(valid_y) - np.min(valid_y) if np.max(valid_y) != np.min(valid_y) else 1
    
    scaled_dx = dx / range_x
    scaled_dy = dy / range_y

    ds = np.sqrt(scaled_dx**2 + scaled_dy**2)
    s = np.concatenate(([0], np.cumsum(ds)))
    
    total_s = s[-1]
    
    # Target uniform steps in s
    target_s = np.linspace(0, total_s, n)
    
    # Find x coords
    chosen_x = np.interp(target_s, s, valid_x)
    
    return [round(float(v), 3) for v in chosen_x]

def main():
    if os.path.exists(MANUAL_FILE):
        with open(MANUAL_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {}

    # Merge extras
    for k, v in extra_formulas.items():
        if k not in data:
            data[k] = []
        # If it's a list of dicts or strings, we just extract strings
        existing_strs = []
        for item in data[k]:
            if isinstance(item, dict):
                if "ground_truth" in item:
                    existing_strs.append(item["ground_truth"]["formula"])
                else:
                    existing_strs.append(item.get("formula", ""))
            else:
                existing_strs.append(item)
                
        for new_f in v:
            if new_f not in existing_strs:
                data[k].append(new_f)

    # Convert everything to the new task format
    new_data = {}
    total = 0
    for k, items in data.items():
        new_data[k] = []
        try:
            a, b, c = map(int, k.split(','))
        except ValueError:
            continue
        config = ComplexityConfig(a, b, c)
        
        for item in items:
            if isinstance(item, dict):
                if "ground_truth" in item:
                    # already in full task format
                    new_data[k].append(item)
                    total += 1
                    continue
                expr_str = item["formula"]
                pts = item.get("x_points")
            else:
                expr_str = item
                pts = None

            if pts is None or len(pts) == 0:
                pts = adaptive_sample(expr_str, n=25)

            # Build full task
            x_sym = sp.Symbol('x', real=True)
            try:
                expr = sp.parse_expr(expr_str, local_dict={'x': x_sym})
                expr = sp.simplify(expr)
                expr_str = str(expr)
            except Exception as e:
                print(f"Skipping {expr_str} due to parse error: {e}")
                continue

            # Calculate safe metadata
            meta_success, metadata, meta_err = DatasetSampler.calculate_metadata_safe(expr, timeout=10)
            if not meta_success:
                print(f"Skipping {expr_str} due to metadata timeout/error: {meta_err}")
                continue
                
            # Calculate safe points based on adaptive sample (pts)
            pts_success, points_res, pts_err = DatasetSampler.calculate_points_safe(expr, n_points=25, custom_x=pts, timeout=10)
            if not pts_success:
                print(f"Skipping {expr_str} due to points timeout/error: {pts_err}")
                continue
                
            x_vals, y_vals = points_res
            
            task = TaskExporter.create_task(expr, x_vals, y_vals, config, metadata)
            task["source"] = "manual"
            
            new_data[k].append(task)
            total += 1
            print(f"Upgraded {k}: {expr_str} -> exact benchmark format")

    with open(MANUAL_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    
    print(f"Done. Upgraded {total} manual functions in {MANUAL_FILE}.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
