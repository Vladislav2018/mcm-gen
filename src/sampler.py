import numpy as np
import sympy as sp
import json
import uuid
from typing import Dict, Any

class TaskExporter:
    """Клас для перетворення згенерованих виразів у формат завдань для LLM."""
    
    @staticmethod
    def create_task(expr: sp.Expr, x: np.ndarray, y: np.ndarray, config: Any) -> Dict[str, Any]:
        """Створює структуру завдання: вхідні дані для LLM та приховані дані для оцінки."""
        task_id = str(uuid.uuid4())[:8]
        
        return {
            "task_id": f"MCM_{config.a}{config.b}{config.c}_{task_id}",
            "complexity_vector": {
                "axis_a_structure": config.a,
                "axis_b_semantics": config.b,
                "axis_c_topology": config.c
            },
            "prompt_data": {
                "description": "Відновіть аналітичну формулу f(x) за наведеними точками даних.",
                "points": [
                    {"x": round(float(xi), 4), "y": round(float(yi), 6)} 
                    for xi, yi in zip(x, y)
                ]
            },
            "ground_truth": {
                "symbolic_expression": str(expr),
                "latex": sp.latex(expr),
                "simplified": str(sp.simplify(expr))
            }
        }