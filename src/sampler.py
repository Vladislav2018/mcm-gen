import numpy as np
import sympy as sp
import uuid
from typing import Dict, Any, Tuple
from sympy.calculus.util import continuous_domain, singularities, periodicity
import multiprocessing

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

def _points_task(expr_str, n, custom_x=None):
    x_sym = sp.Symbol('x', real=True)
    try:
        e = sp.parse_expr(str(expr_str), local_dict={'x': x_sym})
    except:
        raise ValueError("Parse error in worker")

    if custom_x is not None:
        x_vals = np.array(custom_x, dtype=float)
    else:
        intervals = determine_domain(expr_str)
        pts_per_interval = n // len(intervals)
        remainder = n % len(intervals)
        
        xs = []
        for i, (mn, mx) in enumerate(intervals):
            pts = pts_per_interval + (1 if i < remainder else 0)
            xs.append(np.linspace(mn, mx, pts))
        x_vals = np.concatenate(xs)
    
    safe_modules = [
        {'factorial': lambda n: np.clip(np.array(n, dtype=float), 0, 12)}, 
        'numpy'
    ]
    
    f = sp.lambdify(x_sym, e, modules=safe_modules)
    try:
        with np.errstate(all='ignore'):
            y_vals = f(x_vals)
            if np.isscalar(y_vals): y_vals = np.full_like(x_vals, y_vals)
            y_vals = np.array(y_vals, dtype=float)
            
            if not np.all(np.isfinite(y_vals)) or np.any(np.abs(y_vals) > 5000):
                raise ValueError("Values out of bounds")
                
            return x_vals, y_vals
    except Exception as exc:
        raise ValueError(f"Evaluation error: {exc}")

# Wrapper functions for multiprocessing

def _meta_worker(expr_str, conn):
    try:
        res = _meta_task(expr_str)
        conn.send((True, res))
    except Exception as e:
        conn.send((False, str(e)))
    finally:
        conn.close()

def _points_worker(expr_str, n, custom_x, conn):
    try:
        res = _points_task(expr_str, n, custom_x)
        conn.send((True, res))
    except Exception as e:
        conn.send((False, str(e)))
    finally:
        conn.close()

def _generate_worker(a, b, c, conn):
    from .config import ComplexityConfig
    from .generator import ExpressionGenerator
    try:
        config = ComplexityConfig(a, b, c)
        gen = ExpressionGenerator(config)
        expr = gen.generate()
        if expr is None:
            conn.send((False, "Generator returned None after max attempts"))
        else:
            conn.send((True, str(expr)))
    except Exception as e:
        conn.send((False, str(e)))
    finally:
        conn.close()

def _validate_worker(a, b, c, expr_str, conn):
    from .config import ComplexityConfig
    from .validator import TopologyFilter
    import sympy as sp
    try:
        config = ComplexityConfig(a, b, c)
        validator = TopologyFilter(config)
        x = sp.Symbol('x', real=True)
        expr = sp.parse_expr(expr_str, local_dict={'x': x})
        is_valid = validator.check(expr)
        conn.send((True, is_valid))
    except Exception as e:
        conn.send((False, str(e)))
    finally:
        conn.close()

# --- MAIN CLASSES ---

class DatasetSampler:
    
    @staticmethod
    def calculate_metadata_safe(expr: sp.Expr, timeout: int = 5) -> Tuple[bool, Dict, str]:
        parent_conn, child_conn = multiprocessing.Pipe()
        p = multiprocessing.Process(target=_meta_worker, args=(str(expr), child_conn))
        p.start()
        # Close child connection in parent process
        child_conn.close()
        
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join()
            parent_conn.close()
            return False, {"domain": "timeout"}, "Timeout"
            
        if parent_conn.poll():
            success, result = parent_conn.recv()
            parent_conn.close()
            if success:
                return True, result, ""
            else:
                return False, {"domain": "error"}, result
                
        parent_conn.close()
        return False, {"domain": "error"}, "Unknown Error"

    @staticmethod
    def generate_safe(a: int, b: int, c: int, timeout: int = 15) -> Tuple[bool, sp.Expr, str]:
        parent_conn, child_conn = multiprocessing.Pipe()
        p = multiprocessing.Process(target=_generate_worker, args=(a, b, c, child_conn))
        p.start()
        child_conn.close()
        
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join()
            parent_conn.close()
            return False, None, "Timeout"
            
        if parent_conn.poll():
            success, result = parent_conn.recv()
            parent_conn.close()
            if success:
                import sympy as sp
                x = sp.Symbol('x', real=True)
                try:
                    expr = sp.parse_expr(result, local_dict={'x': x})
                    return True, expr, ""
                except Exception as e:
                    return False, None, f"Parse error: {e}"
            else:
                return False, None, str(result)
                
        parent_conn.close()
        return False, None, "Unknown Error"

    @staticmethod
    def validate_safe(a: int, b: int, c: int, expr: sp.Expr, timeout: int = 10) -> Tuple[bool, bool, str]:
        parent_conn, child_conn = multiprocessing.Pipe()
        p = multiprocessing.Process(target=_validate_worker, args=(a, b, c, str(expr), child_conn))
        p.start()
        child_conn.close()
        
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join()
            parent_conn.close()
            return False, False, "Timeout"
            
        if parent_conn.poll():
            success, result = parent_conn.recv()
            parent_conn.close()
            if success:
                return True, result, ""
            else:
                return False, False, str(result)
                
        parent_conn.close()
        return False, False, "Unknown Error"

    @staticmethod
    def calculate_points_safe(expr: sp.Expr, n_points=25, timeout=3, custom_x=None) -> Tuple[bool, Tuple[np.ndarray, np.ndarray], str]:
        parent_conn, child_conn = multiprocessing.Pipe()
        p = multiprocessing.Process(target=_points_worker, args=(str(expr), n_points, custom_x, child_conn))
        p.start()
        child_conn.close()
        
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join()
            parent_conn.close()
            return False, (None, None), "Timeout"
            
        if parent_conn.poll():
            success, result = parent_conn.recv()
            parent_conn.close()
            if success:
                return True, result, ""
            else:
                return False, (None, None), result
                
        parent_conn.close()
        return False, (None, None), "Unknown Error"

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