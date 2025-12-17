import numpy as np
import sympy as sp
import uuid
import json
from typing import Dict, Any, Tuple
# Імпортуємо утиліти для аналізу функцій
from sympy.calculus.util import continuous_domain, singularities, periodicity

class DatasetSampler:
    @staticmethod
    def get_data(expr: sp.Expr, n_points=25) -> Tuple[np.ndarray, np.ndarray, Dict]:
        # Визначаємо змінну x
        symbols = list(expr.free_symbols)
        x_sym = symbols[0] if symbols else sp.Symbol('x')
        
        # --- Аналітичний паспорт функції (PhD Metadata) ---
        metadata = {
            "singularities": [],
            "is_periodic": False,
            "domain": "R"
        }
        
        try:
            # Знаходимо точки розриву
            sings = singularities(expr, x_sym)
            metadata["singularities"] = [str(s) for s in sings if s.is_real]
            
            # Перевіряємо періодичність
            period = periodicity(expr, x_sym)
            metadata["is_periodic"] = period is not None
            if metadata["is_periodic"]:
                metadata["period_value"] = str(period)
            
            # Визначаємо область визначення
            domain = continuous_domain(expr, x_sym, sp.S.Reals)
            metadata["domain"] = str(domain)
        except Exception:
            # Якщо аналіз занадто складний, записуємо unknown
            metadata["domain"] = "complex or unresolvable"

        # --- Генерація точок ---
        # Використовуємо вужчий діапазон для стабільності
        x_vals = np.linspace(-3, 3, n_points)
        
        # Модулі для lambdify з обробкою факторіала
        safe_modules = [
            {'factorial': lambda n: np.clip(np.array(n, dtype=float), 0, 15)}, 
            'numpy'
        ]
        
        try:
            f = sp.lambdify(x_sym, expr, modules=safe_modules)
            with np.errstate(all='ignore'):
                y_vals = f(x_vals)
                if np.isscalar(y_vals): 
                    y_vals = np.full_like(x_vals, y_vals)
                y_vals = np.array(y_vals, dtype=float)
                
                # Валідація результатів
                if not np.all(np.isfinite(y_vals)) or np.any(np.abs(y_vals) > 1e6):
                    return None, None, None
                    
                return x_vals, y_vals, metadata
        except:
            return None, None, None

class TaskExporter:
    """Клас для пакування результатів у формат JSON."""
    @staticmethod
    def create_task(expr: sp.Expr, x: np.ndarray, y: np.ndarray, config: Any, metadata: Dict) -> Dict[str, Any]:
        task_id = str(uuid.uuid4())[:8]
        return {
            "task_id": f"MCM_{config.a}{config.b}{config.c}_{task_id}",
            "complexity_vector": {
                "axis_a_structure": config.a,
                "axis_b_semantics": config.b,
                "axis_c_topology": config.c
            },
            "prompt_data": {
                "description": "Identify the analytical form of the function f(x) from the points.",
                "points": [{"x": round(float(xi), 4), "y": round(float(yi), 6)} for xi, yi in zip(x, y)]
            },
            "ground_truth": {
                "symbolic_expression": str(expr),
                "latex": sp.latex(expr),
                "analytical_properties": metadata
            }
        }