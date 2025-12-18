import numpy as np
import sympy as sp
import uuid
from typing import Dict, Any, Tuple
from sympy.calculus.util import continuous_domain, singularities, periodicity
from src.utils import run_with_timeout

# --- WORKER FUNCTIONS (Must be at module level for Windows multiprocessing) ---

def _meta_task(e):
    """Обчислення метаданих (виконується в окремому процесі)."""
    symbols = list(e.free_symbols)
    x_sym = symbols[0] if symbols else sp.Symbol('x')
    meta = {"singularities": [], "is_periodic": False, "domain": "R"}
    
    # Periodicity
    try:
        period = periodicity(e, x_sym)
        meta["is_periodic"] = period is not None
        if meta["is_periodic"]: meta["period_value"] = str(period)
    except: pass 

    # Singularities
    try:
        sings = singularities(e, x_sym)
        if isinstance(sings, sp.FiniteSet):
            meta["singularities"] = [str(s) for s in sings if s.is_real]
        else:
            meta["singularities"] = ["complex_or_infinite"]
    except: 
        meta["singularities"] = ["analysis_error"]

    # Domain
    try:
        domain = continuous_domain(e, x_sym, sp.S.Reals)
        meta["domain"] = str(domain)
    except:
        meta["domain"] = "analysis_timeout"
    
    return meta

def _points_task(e, n):
    """Обчислення точок (виконується в окремому процесі)."""
    symbols = list(e.free_symbols)
    x_sym = symbols[0] if symbols else sp.Symbol('x')
    x_vals = np.linspace(-3, 3, n)
    
    # Обмеження для факторіалів
    safe_modules = [
        {'factorial': lambda n: np.clip(np.array(n, dtype=float), 0, 12)}, 
        'numpy'
    ]
    
    f = sp.lambdify(x_sym, e, modules=safe_modules)
    with np.errstate(all='ignore'): # Ігноруємо попередження про sqrt(-1) тощо
        y_vals = f(x_vals)
        if np.isscalar(y_vals): y_vals = np.full_like(x_vals, y_vals)
        y_vals = np.array(y_vals, dtype=float)
        
        # Перевірка на валідність
        if not np.all(np.isfinite(y_vals)) or np.any(np.abs(y_vals) > 5000):
            raise ValueError("Values out of bounds or non-finite")
            
        return x_vals, y_vals

# --- MAIN CLASSES ---

class DatasetSampler:
    
    @staticmethod
    def calculate_metadata_safe(expr: sp.Expr, timeout: int = 5) -> Tuple[bool, Dict, str]:
        """Обчислює метадані з тайм-аутом."""
        # Передаємо глобальну функцію _meta_task, а не локальну
        success, result = run_with_timeout(_meta_task, (expr,), timeout)
        if success:
            return True, result, ""
        else:
            return False, {"domain": "timeout_during_analysis"}, result

    @staticmethod
    def calculate_points_safe(expr: sp.Expr, n_points=25, timeout: int = 3) -> Tuple[bool, Tuple[np.ndarray, np.ndarray], str]:
        """Обчислює точки графіка з тайм-аутом."""
        # Передаємо глобальну функцію _points_task
        success, result = run_with_timeout(_points_task, (expr, n_points), timeout)
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
            "prompt_data": {
                "points": points_data
            },
            "ground_truth": {
                "formula": str(expr),
                "latex": sp.latex(expr),
                "properties": metadata
            }
        }