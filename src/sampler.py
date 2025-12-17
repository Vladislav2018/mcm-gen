import numpy as np
import sympy as sp

class DatasetSampler:
    """Генерація безшумних даних на основі формули[cite: 162]."""
    @staticmethod
    def get_data(expr: sp.Expr, x_range=(-10, 10), n_points=500):
        f = sp.lambdify(list(expr.free_symbols)[0], expr, 'numpy')
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = f(x)
        return x, y