import numpy as np
import sympy as sp
import uuid
from typing import Dict, Any, Tuple
from sympy.calculus.util import continuous_domain, singularities, periodicity
from src.utils import run_with_timeout

# --- WORKER FUNCTIONS ---

def _meta_task(expr_str):
    """
    Приймає рядок формули (щоб уникнути проблем піклінгу складних об'єктів),
    парсить його з real=True і аналізує.
    """
    x = sp.Symbol('x', real=True)
    try:
        e = sp.parse_expr(str(expr_str), local_dict={'x': x})
    except:
        return {"error": "parse_error"}

    meta = {"singularities": [], "is_periodic": False, "domain": "R"}
    
    # Periodicity
    try:
        period = periodicity(e, x)
        meta["is_periodic"] = period is not None
        if meta["is_periodic"]: meta["period_value"] = str(period)
    except: pass 

    # Singularities
    try:
        sings = singularities(e, x)
        
        # SymPy часто повертає EmptySet, що добре
        if sings is sp.S.EmptySet:
            meta["singularities"] = []
        elif isinstance(sings, sp.FiniteSet):
            meta["singularities"] = [str(s) for s in sings]
        else:
            # Для поліномів іноді буває дивна поведінка, спробуємо solve знаменника
            numer, denom = sp.fraction(sp.together(e))
            if denom != 1:
                roots = sp.solve(denom, x)
                if roots:
                    meta["singularities"] = [str(r) for r in roots]
                else:
                    meta["singularities"] = []
            else:
                 meta["singularities"] = []
    except: 
        meta["singularities"] = ["analysis_error"]

    # Domain
    try:
        domain = continuous_domain(e, x, sp.S.Reals)
        meta["domain"] = str(domain)
    except:
        meta["domain"] = "analysis_timeout"
    
    return meta

def _points_task(expr_str, n):
    x_sym = sp.Symbol('x', real=True)
    try:
        e = sp.parse_expr(str(expr_str), local_dict={'x': x_sym})
    except:
        raise ValueError("Parse error in worker")

    x_vals = np.linspace(-3, 3, n)
    
    safe_modules = [
        {'factorial': lambda n: np.clip(np.array(n, dtype=float), 0, 12)}, 
        'numpy'
    ]
    
    f = sp.lambdify(x_sym, e, modules=safe_modules)
    with np.errstate(all='ignore'):
        y_vals = f(x_vals)
        if np.isscalar(y_vals): y_vals = np.full_like(x_vals, y_vals)
        y_vals = np.array(y_vals, dtype=float)
        
        if not np.all(np.isfinite(y_vals)) or np.any(np.abs(y_vals) > 5000):
            raise ValueError("Values out of bounds")
            
        return x_vals, y_vals

# --- MAIN CLASSES ---

class DatasetSampler:
    
    @staticmethod
    def calculate_metadata_safe(expr: sp.Expr, timeout: int = 5) -> Tuple[bool, Dict, str]:
        # Передаємо рядок, а не об'єкт SymPy
        success, result = run_with_timeout(_meta_task, (str(expr),), timeout)
        if success:
            return True, result, ""
        else:
            return False, {"domain": "timeout"}, result

    @staticmethod
    def calculate_points_safe(expr: sp.Expr, n_points=25, timeout: int = 3) -> Tuple[bool, Tuple[np.ndarray, np.ndarray], str]:
        success, result = run_with_timeout(_points_task, (str(expr), n_points), timeout)
        if success:
            return True, result, ""
        else:
            return False, (None, None), result

class TaskExporter:
    @staticmethod
    def create_task(expr: sp.Expr, x: np.ndarray, y: np.ndarray, config: Any, metadata: Dict) -> Dict[str, Any]:
        task_id = str(uuid.uuid4())[:8]
        points_data = []
        if x is not None and y is not None:
             points_data = [{"x": round(float(xi), 3), "y": round(float(yi), 4)} for xi, yi in zip(x, y)]

        return {
            "task_id": f"MCM_{config.a}{config.b}{config.c}_{task_id}",
            "complexity_vector": {"a": config.a, "b": config.b, "c": config.c},
            "prompt_data": {"points": points_data},
            "ground_truth": {
                "formula": str(expr),
                "latex": sp.latex(expr),
                "properties": metadata
            }
        }