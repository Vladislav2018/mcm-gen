import sympy as sp
import numpy as np
from .config import ComplexityConfig

class TopologyFilter:
    """Топологічна валідація згідно з Axis C[cite: 132]."""
    def __init__(self, config: ComplexityConfig):
        self.config = config

    def check(self, expr: sp.Expr) -> bool:
        if self.config.c == 0: return self._is_regular(expr)
        if self.config.c == 2: return self._has_asymptotes(expr) # [cite: 100]
        # Додаткові перевірки C1 та C3 додаються тут
        return True

    def _is_regular(self, expr: sp.Expr) -> bool:
        """Перевірка на гладкість (C0)[cite: 94]."""
        f = sp.lambdify(self.config.x, expr, 'numpy')
        test_vals = np.linspace(-5, 5, 50)
        try:
            return np.all(np.isfinite(f(test_vals)))
        except: return False

    def _has_asymptotes(self, expr: sp.Expr) -> bool:
        """Пошук полюсів (C2)[cite: 101]."""
        _, den = sp.fraction(expr)
        return den != 1 and len(sp.solve(den, self.config.x)) > 0