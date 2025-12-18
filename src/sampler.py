import numpy as np
import sympy as sp
import uuid
import json
from typing import Dict, Any, Tuple
from sympy.calculus.util import continuous_domain, singularities, periodicity

class DatasetSampler:
    @staticmethod
    def get_data(expr: sp.Expr, n_points=25) -> Tuple[np.ndarray, np.ndarray, Dict]:
        symbols = list(expr.free_symbols)
        x_sym = symbols[0] if symbols else sp.Symbol('x')
        
        metadata = {
            "singularities": [],
            "is_periodic": False,
            "domain": "R"
        }
        
        # Блок безпечного аналізу (PhD Metadata)
        try:
            # Перевіряємо періодичність (зазвичай працює швидко)
            period = periodicity(expr, x_sym)
            metadata["is_periodic"] = period is not None
            if metadata["is_periodic"]: metadata["period_value"] = str(period)

            # БЕЗПЕЧНИЙ ПОШУК РОЗРИВІВ
            # Використовуємо спрощений підхід, щоб не зациклити SymPy
            sings = singularities(expr, x_sym)
            # Обробляємо тільки скінченні множини точок (FiniteSet)
            if isinstance(sings, sp.FiniteSet):
                metadata["singularities"] = [str(s) for s in sings if s.is_real]
            else:
                metadata["singularities"] = ["non-finite or complex set"]

            # Область визначення
            domain = continuous_domain(expr, x_sym, sp.S.Reals)
            metadata["domain"] = str(domain)
        except Exception:
            # Якщо аналіз «завис» або видав помилку — просто ігноруємо метадані
            metadata["domain"] = "analysis_timeout"

        # --- Генерація точок (Оптимізовано для ресурсів) ---
        x_vals = np.linspace(-3, 3, n_points)
        
        # Обмеження для факторіалів та експонент, щоб не було Overflow
        safe_modules = [
            {'factorial': lambda n: np.clip(np.array(n, dtype=float), 0, 12)}, 
            'numpy'
        ]
        
        try:
            f = sp.lambdify(x_sym, expr, modules=safe_modules)
            with np.errstate(all='ignore'):
                y_vals = f(x_vals)
                if np.isscalar(y_vals): y_vals = np.full_like(x_vals, y_vals)
                y_vals = np.array(y_vals, dtype=float)
                
                # Жорсткий фільтр значень (щоб LLM не бачила гігантських чисел)
                if not np.all(np.isfinite(y_vals)) or np.any(np.abs(y_vals) > 5000):
                    return None, None, None
                    
                return x_vals, y_vals, metadata
        except:
            return None, None, None

class TaskExporter:
    @staticmethod
    def create_task(expr: sp.Expr, x: np.ndarray, y: np.ndarray, config: Any, metadata: Dict) -> Dict[str, Any]:
        task_id = str(uuid.uuid4())[:8]
        return {
            "task_id": f"MCM_{config.a}{config.b}{config.c}_{task_id}",
            "complexity_vector": {"a": config.a, "b": config.b, "c": config.c},
            "prompt_data": {
                "points": [{"x": round(float(xi), 3), "y": round(float(yi), 4)} for xi, yi in zip(x, y)]
            },
            "ground_truth": {
                "formula": str(expr),
                "latex": sp.latex(expr),
                "properties": metadata
            }
        }