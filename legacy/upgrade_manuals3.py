import json
import numpy as np
import sympy as sp
import os
import multiprocessing
from src.config import ComplexityConfig
from src.sampler import DatasetSampler, TaskExporter

MANUAL_FILE = "manual_formulas.json"

extra_formulas = {
    "0,0,0": ["5*x - 2", "-x + 7.5", "10", "4*x + 1", "0", "x + 2", "10*x - 3", "-0.5*x + 1", "3"],
    "0,1,0": ["sin(x) + 2", "cos(x) - 1", "exp(x+1)", "sin(x) + cos(x)", "log(exp(x)+1)"],
    "0,2,0": ["abs(x) + 2", "abs(x - 3)", "Piecewise((x, x > 1), (-1, True))", "floor(x)", "x * abs(x)"],
    "0,3,0": ["erf(x) + 1", "besselj(0, abs(x))", "zeta(abs(x)+2)", "erf(x)*x"],
    "1,0,0": ["x**2 + x + 1", "(x-1)**2", "-2*x**3 + 4", "0.5*x**2 - 2*x", "x**3 - 3*x", "0.2*x**4"],
    "1,1,0": ["sin(x**2)", "exp(-x**2)", "cos(x)*exp(-0.2*x)", "sin(x) + cos(2*x)", "log(x**2 + 1)", "sin(x)**2 + cos(x)"],
    "1,2,0": ["abs(x**2 - 1)", "x * floor(x/2)", "Piecewise((x**2, x > 0), (-x, True))", "abs(x) * x**2", "abs(x+1) + abs(x-1)"],
    "1,3,0": ["erf(x**2)", "gamma(abs(x)+2) - x", "zeta(abs(x)+1.5)*x", "besselj(1, abs(x)) * x"],
    "2,0,0": ["x**4 - 2*x**2 + x", "(x**2 - 1)**2", "0.1*x**5 - x**3", "x*(x-1)*(x+2)"],
    "2,1,0": ["sin(exp(x))", "exp(sin(x))", "sin(x**2 + x)", "cos(x)*exp(-x**2)"],
    "2,2,0": ["abs(x**3 - x)", "floor(exp(x/2))", "Piecewise((x**2, x > 1), (x, x > 0), (0, True))", "abs(x**2 - x) - x"],
    "2,3,0": ["erf(sin(x))", "besselj(0, abs(x**2))", "gamma(abs(sin(x))+1)"],
    "3,0,0": ["x**5 - 3*x**4 + 2*x**2", "0.1*x**6 + 2*x", "(x**2-4)*(x**2-1)*x"],
    "3,1,0": ["sin(x)*cos(x**2) - exp(-x)", "log(cosh(x)) + log(2)"],
    "3,2,0": ["Abs(x**3 - x) + 2", "Piecewise((sin(x), x>0), (x**2, True))"],
    "3,3,0": ["erf(x)*exp(-x**2)", "gamma(abs(x)+1)*sin(x)"],
    "0,0,1": ["cos(x)", "sin(x+1)"],
    "1,0,1": ["sin(3*x)", "cos(0.5*x)"],
    "2,0,1": ["sin(x)*cos(2*x)", "sin(x)**2"],
    "3,0,1": ["sin(x)**3 - cos(x)", "sin(cos(x))"],
    "0,0,2": ["1/x", "1/(x+1)", "1/(x-2)", "x/(x-1)", "2/(x+0.5)"],
    "1,0,2": ["1/(x**2 - 1)", "x**2/(x - 3)", "1/(x**2 + x)", "x/(x**2 - 4)"],
    "2,0,2": ["(x**3 + 1)/(x**2 - 4)", "x**3/(x - 1)", "1/(x**3 - x)"],
    "3,0,2": ["x**5/(x - 0.1)", "(x**3 - 1)/(x**2 - x)"],
    "0,1,2": ["tan(x)", "1/cos(x)", "exp(1/x)"],
    "1,1,2": ["tan(2*x)", "1/sin(x)", "exp(x)/(x-2)"],
    "2,1,2": ["sin(x)/(cos(x) - 0.5)", "log(abs(x-1))"],
    "3,1,2": ["tan(exp(x))", "1/sin(x**3)"],
    "0,2,2": ["1/abs(x-0.5)", "floor(x)/(x-1)"],
    "1,2,2": ["abs(x)/(x-1)", "1/abs(x**2 - 1)"],
    "2,2,2": ["factorial(floor(abs(x)))/(x+2)", "abs(x**2 - 4)/(x-2)"],
    "3,2,2": ["floor(exp(x))/(x-2.5)", "Piecewise((1/x, x > 0.1), (0, True))"],
    "0,3,2": ["1/erf(x)", "gamma(x)/x"],
    "1,3,2": ["besselj(0, x)/(x-0.5)", "erf(x)/(x**2 - 1)"],
    "2,3,2": ["1/gamma(x)", "zeta(1/(x-1))"],
    "3,3,2": ["gamma(1/x)*x"],
    "0,0,3": ["x**(-0.5)", "(x-1)**(-1.5)"],
    "1,0,3": ["(x**2 - 1)**(-0.5)", "x**(-1.5)"],
    "2,0,3": ["(x**3 - 2)**(-0.5)"],
    "3,0,3": ["(x**4 - 1)**(-0.25)"],
    "0,1,3": ["sin(1/x)", "cos(1/(x-1))"],
    "1,1,3": ["sin(1/x**2)"],
    "2,1,3": ["exp(1/x)*sin(1/x)"],
    "3,1,3": ["sin(1/(x**3 - x))"],
    "0,2,3": ["abs(x)**(-0.5)", "Piecewise((1/x, x!=0), (10, True))"],
    "1,2,3": ["abs(x**2 - 1)**(-0.5)"],
    "2,2,3": ["abs(x**3 - x)**(-0.5)"],
    "3,2,3": ["Piecewise((sin(1/x), x!=0), (0, True))"],
    "0,3,3": ["gamma(x)*sin(1/x)"],
    "1,3,3": ["zeta(x)*log(abs(x))"],
    "2,3,3": ["erf(1/x)"],
    "3,3,3": ["besselj(0, 1/x)"]
}

def adaptive_sample(expr_str, n=25):
    x_sym = sp.Symbol('x', real=True)
    try:
        expr = sp.parse_expr(expr_str, local_dict={'x': x_sym})
        safe_modules = [{'factorial': lambda n: np.clip(np.array(n, dtype=float), 0, 12)}, "scipy", "numpy"]
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

    valid_mask = np.isfinite(y) & (np.abs(y) < 1000)
    if not np.any(valid_mask):
        return list(np.round(np.linspace(-3, 3, n), 3))

    valid_x = dense_x[valid_mask]
    valid_y = y[valid_mask]

    if len(valid_x) < n:
        return [round(float(v), 3) for v in valid_x]

    dx = np.diff(valid_x)
    dy = np.diff(valid_y)
    
    range_x = np.max(valid_x) - np.min(valid_x)
    range_y = np.max(valid_y) - np.min(valid_y) if np.max(valid_y) != np.min(valid_y) else 1
    
    if range_x == 0: range_x = 1
    
    scaled_dx = dx / range_x
    scaled_dy = dy / range_y

    ds = np.sqrt(scaled_dx**2 + scaled_dy**2)
    s = np.concatenate(([0], np.cumsum(ds)))
    
    total_s = s[-1]
    target_s = np.linspace(0, total_s, n)
    chosen_x = np.interp(target_s, s, valid_x)
    
    return [round(float(v), 3) for v in chosen_x]

def get_points(expr, custom_x=None, n=25):
    x_sym = sp.Symbol('x', real=True)
    if custom_x is not None:
        x_vals = np.array(custom_x, dtype=float)
    else:
        x_vals = np.linspace(-3, 3, n)
        
    try:
        safe_modules = [{'factorial': lambda n: np.clip(np.array(n, dtype=float), 0, 12)}, 'numpy']
        f = sp.lambdify(x_sym, expr, modules=safe_modules)
        with np.errstate(all='ignore'):
            y_vals = f(x_vals)
            if np.isscalar(y_vals): y_vals = np.full_like(x_vals, y_vals)
            y_vals = np.array(y_vals, dtype=float)
    except Exception:
        try:
            y_vals = []
            for val in x_vals:
                y_eval = expr.subs(x_sym, float(val)).evalf()
                y_vals.append(float(y_eval))
            y_vals = np.array(y_vals, dtype=float)
        except Exception:
            return False, None, None

    if not np.all(np.isfinite(y_vals)) or np.any(np.abs(y_vals) > 5000) or np.any(np.iscomplex(y_vals)):
        return False, None, None
    return True, x_vals, y_vals

def dumps_compact(data):
    lines = ["{"]
    sorted_keys = sorted(data.keys())
    for i, k in enumerate(sorted_keys):
        tasks = data[k]
        lines.append(f'  "{k}": [')
        for j, task in enumerate(tasks):
            lines.append('    {')
            lines.append(f'      "task_id": "{task["task_id"]}",')
            lines.append(f'      "complexity_vector": {json.dumps(task["complexity_vector"])},')
            
            pts = task["prompt_data"]["points"]
            pts_str = "[" + ", ".join([json.dumps(p) for p in pts]) + "]"
            lines.append(f'      "prompt_data": {{"points": {pts_str}}},')
            
            lines.append(f'      "ground_truth": {json.dumps(task["ground_truth"])},')
            lines.append(f'      "source": "{task.get("source", "manual")}"')
            
            if j < len(tasks) - 1:
                lines.append('    },')
            else:
                lines.append('    }')
        if i < len(sorted_keys) - 1:
            lines.append('  ],')
        else:
            lines.append('  ]')
    lines.append("}")
    return "\n".join(lines)

def main():
    if os.path.exists(MANUAL_FILE):
        with open(MANUAL_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {}

    for k, v in extra_formulas.items():
        if k not in data:
            data[k] = []
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
            try:
                parsed_new = sp.simplify(sp.parse_expr(new_f, local_dict={'x': sp.Symbol('x')}))
            except:
                parsed_new = new_f
            
            already_have = False
            for ex in existing_strs:
                try:
                    if sp.simplify(sp.parse_expr(ex, local_dict={'x': sp.Symbol('x')})) == parsed_new:
                        already_have = True
                        break
                except:
                    if ex == new_f:
                        already_have = True
            
            if not already_have:
                data[k].append(new_f)

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
                    expr_str = item["ground_truth"]["formula"]
                    pts = [p["x"] for p in item["prompt_data"]["points"]]
                else:
                    expr_str = item["formula"]
                    pts = item.get("x_points")
            else:
                expr_str = item
                pts = None

            x_sym = sp.Symbol('x', real=True)
            try:
                expr = sp.parse_expr(expr_str, local_dict={'x': x_sym})
                expr = sp.simplify(expr)
                expr_str = str(expr)
            except Exception as e:
                print(f"Skipping {expr_str} due to parse error: {e}")
                continue

            if pts is None or len(pts) == 0:
                pts = adaptive_sample(expr_str, n=25)

            meta_success, metadata, meta_err = DatasetSampler.calculate_metadata_safe(expr, timeout=10)
            if not meta_success:
                print(f"Skipping {expr_str} due to metadata timeout/error: {meta_err}")
                continue
                
            pts_success, x_vals, y_vals = get_points(expr, custom_x=pts, n=25)
            if not pts_success:
                print(f"Skipping {expr_str} due to points timeout/error")
                continue
            
            task = TaskExporter.create_task(expr, x_vals, y_vals, config, metadata)
            task["source"] = "manual"
            
            new_data[k].append(task)
            total += 1
            print(f"Upgraded {k}: {expr_str}")

    with open(MANUAL_FILE, 'w', encoding='utf-8') as f:
        f.write(dumps_compact(new_data))
    
    print(f"Done. Upgraded {total} manual functions in {MANUAL_FILE}. All points are inline!")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
