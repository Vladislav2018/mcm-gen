import json
import os
import numpy as np
import sympy as sp
from sympy.calculus.util import continuous_domain, periodicity
import multiprocessing
from src.config import ComplexityConfig
from src.sampler import TaskExporter

MANUAL_FILE = "manual_formulas.json"

def get_base_formulas(a, b, c):
    # Generates 3 unique strings based on a,b,c
    # a: depth/complexity. 0=simple, 1=quadratic, 2=cubic, 3=quartic/poly
    # b: ops. 0=alg, 1=+exp/trig, 2=+abs/Piecewise, 3=+erf/besselj
    # c: topo. 0=reg, 1=periodic, 2=asymptotes, 3=chaos/singular
    f1, f2, f3 = "", "", ""
    
    # Inner term based on A
    if a == 0:
        i1, i2, i3 = "x", "0.5*x", "(x+1)"
        p1, p2, p3 = "2*x", "x+2", "(x-1)"
    elif a == 1:
        i1, i2, i3 = "x**2", "(x**2-1)", "(0.5*x**2+x)"
        p1, p2, p3 = "(x**2+1)", "x**2-x", "0.2*x**2"
    elif a == 2:
        i1, i2, i3 = "x**3", "(x**3-x)", "(x**3+2*x**2)"
        p1, p2, p3 = "0.1*x**3", "(x**3+1)", "(x**2*x)"
    else:
        i1, i2, i3 = "x**4", "(x**4-2*x**2)", "(x**4+x**3)"
        p1, p2, p3 = "x**4-x", "0.1*x**4", "(x**2+1)**2"

    # Core operation based on B
    if b == 0:
        op1, op2, op3 = lambda x: f"{x}", lambda x: f"-{x}", lambda x: f"2*{x}"
    elif b == 1:
        op1, op2, op3 = lambda x: f"exp({x}*0.1)", lambda x: f"log(abs({x})+1)", lambda x: f"sin({x})*exp(-0.1*{x})"
    elif b == 2:
        op1, op2, op3 = lambda x: f"Abs({x})", lambda x: f"floor({x})", lambda x: f"Piecewise(({x}, x>0), (-{x}, True))"
    else:
        op1, op2, op3 = lambda x: f"erf({x})", lambda x: f"gamma(Abs({x})+1.5)", lambda x: f"besselj(0, Abs({x}))"

    # Topology envelope based on C
    if c == 0: # regular
        f1, f2, f3 = op1(i1), op2(i2), op3(i3)
    elif c == 1: # periodic
        f1 = f"sin({op1(i1)})"
        f2 = f"cos({op2(i2)})"
        if b == 0: # force periodic envelope
            f3 = f"sin({p3})"
        elif b >= 1:
            f3 = f"tan({p3})*cos({p3})" # might be periodic
        else:
            f3 = f"sin({op3(i3)})"
    elif c == 2: # asymptote
        d1, d2, d3 = "(x-0.1)", "(x+0.2)", "(x-1.1)"
        f1 = f"({op1(i1)})/{d1}"
        f2 = f"({op2(i2)})/{d2}"
        f3 = f"({op3(i3)})/{d3}"
    elif c == 3: # singular/chaos
        f1 = f"{op1(i1)}*x**5"
        f2 = f"{op2(i2)}*exp(abs(x))"
        f3 = f"Abs({op3(i3)})**0.1 + x**5"
        
    return [f1, f2, f3]

extra_formulas = {}
for a in range(4):
    for b in range(4):
        for c in range(4):
            extra_formulas[f"{a},{b},{c}"] = get_base_formulas(a, b, c)

# Specific overrides for tricky classes that sympy struggles with natively:
extra_formulas["0,0,0"] = ["5*x + 2", "x - 4", "10*x - 3", "-0.5*x + 1", "3", "x"]
extra_formulas["0,1,0"] = ["sin(x)", "exp(x)", "-x + cos(x)", "sin(x) + cos(x)", "exp(-x)"]
extra_formulas["0,2,0"] = ["Abs(x)", "Piecewise((x, x > 0), (-x, True))", "Abs(x - 2)"]
extra_formulas["0,3,0"] = ["x*erf(x)", "erf(x) + 1", "x*erf(x-1)"]
extra_formulas["0,0,1"] = ["sin(x)", "cos(x+1)"]
extra_formulas["0,0,2"] = ["1/(x-0.5)", "1/(x+0.5)", "x/(x-1.5)"]
extra_formulas["3,3,3"] = ["erf(x)*x**5", "gamma(Abs(x)+1.5)*x**4", "besselj(0, Abs(x))*x**5"]

def adaptive_sample(expr_str, n=25):
    x_sym = sp.Symbol('x', real=True)
    try:
        expr = sp.parse_expr(expr_str, local_dict={'x': x_sym})
        safe_modules = [{'factorial': lambda n: np.clip(np.array(n, dtype=float), 0, 12)}, "scipy", "numpy"]
        f = sp.lambdify(x_sym, expr, modules=safe_modules)
    except:
        return list(np.round(np.linspace(-3, 3, n), 3))

    intervals = determine_domain(expr_str)
    # Create a dense grid to find curvature
    dense_x = np.concatenate([np.linspace(m, M, 1000//len(intervals)) for m, M in intervals])
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

def get_points(expr, custom_x=None, n=25, x_min=-3.0, x_max=3.0):
    """Safely evaluates points, first trying fast numpy lambdify, 
    then falling back to sympy evalf if scipy specials fail.
    Skips if values are out of bounds or nan."""
    x_sym = list(expr.free_symbols)[0] if expr.free_symbols else sp.Symbol('x')
    
    if custom_x is not None:
        xs = np.array(custom_x, dtype=float)
    else:
        intervals = determine_domain(str(expr))
        pts_per_interval = n // len(intervals)
        remainder = n % len(intervals)
        
        xs = []
        for i, (mn, mx) in enumerate(intervals):
            pts = pts_per_interval + (1 if i < remainder else 0)
            xs.append(np.linspace(mn, mx, pts))
        xs = np.concatenate(xs)
        
    try:
        safe_modules = [{'factorial': lambda n: np.clip(np.array(n, dtype=float), 0, 12)}, 'scipy', 'numpy']
        f = sp.lambdify(x_sym, expr, modules=safe_modules)
        with np.errstate(all='ignore'):
            y_vals = f(xs)
            if np.isscalar(y_vals): y_vals = np.full_like(xs, y_vals)
            y_vals = np.array(y_vals, dtype=float)
    except Exception:
        try:
            y_vals = []
            for val in xs:
                y_eval = expr.subs(x_sym, float(val)).evalf()
                y_vals.append(float(y_eval))
            y_vals = np.array(y_vals, dtype=float)
        except Exception:
            return False, None, None

    if not np.all(np.isfinite(y_vals)) or np.any(np.abs(y_vals) > 50000) or np.any(np.iscomplex(y_vals)):
        return False, None, None
    
    if len(xs) >= n * 0.8:
        return True, list(xs[:n]), list(y_vals[:n])
    return False, [], []

def calculate_metadata_safe(expr, timeout=10):
    x = sp.Symbol('x', real=True)
    meta = {"singularities": [], "is_periodic": False, "domain": "R"}
    try:
        period = periodicity(expr, x)
        meta["is_periodic"] = period is not None
        if meta["is_periodic"]: meta["period_value"] = str(period)
    except: pass 
    try:
        domain = continuous_domain(expr, x, sp.S.Reals)
        meta["domain"] = str(domain)
    except:
        meta["domain"] = "Reals"
    return True, meta, ""

import sympy as sp
import random
import uuid

def determine_domain(expr_str):
    """Heuristic to pick a good sampling domain based on operators.
    Returns a LIST of tuples: [(min1, max1), (min2, max2), ...]"""
    expr_str = expr_str.lower()
    
    if '/' in expr_str and '(x' in expr_str:
        return [(-3.0, -0.1), (0.1, 3.0)]

    # Rapid growth: tight bounds
    if 'exp' in expr_str or 'gamma' in expr_str or '**x' in expr_str or 'x**5' in expr_str:
        return [(-1.5, 1.5)]
        
    # High frequency / singularities: tight bounds around origin
    if 'sin(1/' in expr_str or 'cos(1/' in expr_str:
        return [(-1.0, -0.05), (0.05, 1.0)]
        
    # Discontinuous / Sign functions
    if 'sign' in expr_str or 'abs' in expr_str:
        return [(-3.0, -0.05), (0.05, 3.0)]
        
    # Normal trig: wide enough to see periods, but not too wide
    if 'sin' in expr_str or 'cos' in expr_str:
        return [(-4.0, 4.0)] # ~ 1.25 periods of 2pi
        
    # Default for polynomials and well-behaved functions
    return [(-3.0, 3.0)]

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
        import shutil
        # keep backup
        shutil.copy(MANUAL_FILE, MANUAL_FILE + ".bak")

    new_data = {}
    import uuid
    for k, v in extra_formulas.items():
        new_data[k] = []
        try:
            a, b, c = map(int, k.split(','))
        except ValueError:
            continue
        config = ComplexityConfig(a, b, c)
        
        for expr_str in v:
            x_sym = sp.Symbol('x', real=True)
            try:
                expr = sp.parse_expr(expr_str, local_dict={'x': x_sym})
                expr = sp.simplify(expr)
                expr_str = str(expr)
            except Exception as e:
                continue

            x_min, x_max = determine_domain(expr_str)[0] # Just for scoping compatibility
            
            pts = adaptive_sample(expr_str, n=25)
            meta_success, metadata, meta_err = calculate_metadata_safe(expr, timeout=10)
            pts_success, x_vals, y_vals = get_points(expr, custom_x=pts, n=25)
            
            if not pts_success:
                # Retry with basic evenly spaced intervals if adaptive failed
                pts_success, x_vals, y_vals = get_points(expr, custom_x=None, n=25)
            if not pts_success:
                # Skip if absolutely impossible to evaluate (out of bounds)
                continue
            
            task_id = str(uuid.uuid4())[:8]
            points_data = [{"x": round(float(xi), 3), "y": round(float(yi), 4)} for xi, yi in zip(x_vals, y_vals)]
            
            task = {
                "task_id": f"MCM_{config.a}{config.b}{config.c}_{task_id}",
                "complexity_vector": {"a": config.a, "b": config.b, "c": config.c},
                "prompt_data": {"points": points_data},
                "ground_truth": {
                    "formula": str(expr),
                    "latex": sp.latex(expr),
                    "properties": metadata
                },
                "source": "manual"
            }
            new_data[k].append(task)
            print(f"Added {k}: {expr_str}")

    with open(MANUAL_FILE, 'w', encoding='utf-8') as f:
        f.write(dumps_compact(new_data))
    
    total = sum(len(l) for l in new_data.values())
    print(f"Done. Built {total} manual functions in {MANUAL_FILE}. All points are inline!")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
