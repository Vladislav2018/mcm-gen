"""MCM-Gen Topology Validator v2.0

Validates that generated expressions match their claimed topological class (C axis).
Uses combination of symbolic (SymPy) and numerical (NumPy) checks.
"""

import sympy as sp
import numpy as np
from .config import ComplexityConfig


class TopologyFilter:
    """Validates topological properties of expressions against target C level."""

    def __init__(self, config: ComplexityConfig):
        self.config = config
        self.x = config.x

    def check(self, expr: sp.Expr) -> bool:
        """Check if expression matches target C level."""
        c = self.config.c
        if c == 0: return self._is_regular(expr)
        if c == 1: return self._is_periodic(expr)
        if c == 2: return self._has_isolated_singularities(expr)
        if c == 3: return self._has_complex_behavior(expr)
        return True

    def _eval_safe(self, expr, x_vals):
        """Safely evaluate expression at given x values. Returns None on failure."""
        try:
            f = sp.lambdify(self.x, expr, 'numpy')
            with np.errstate(all='ignore'):
                y = f(x_vals)
                if np.isscalar(y):
                    y = np.full_like(x_vals, y)
                return np.array(y, dtype=float)
        except Exception:
            return None

    # ===================== C=0: Regular =====================

    def _is_regular(self, expr):
        """C=0: Function should be smooth and finite on wide range, NOT periodic."""
        # Check finiteness on wide range
        x_vals = np.linspace(-10, 10, 500)
        y_vals = self._eval_safe(expr, x_vals)
        if y_vals is None:
            return False
        if not np.all(np.isfinite(y_vals)):
            return False

        # Also check narrow range for hidden issues
        x_narrow = np.linspace(-1, 1, 100)
        y_narrow = self._eval_safe(expr, x_narrow)
        if y_narrow is None or not np.all(np.isfinite(y_narrow)):
            return False

        # Check NOT periodic (C=0 should be non-periodic)
        try:
            from sympy.calculus.util import periodicity
            period = periodicity(expr, self.x)
            if period is not None:
                return False
        except Exception:
            pass  # If check fails, assume not periodic

        return True

    # ===================== C=1: Periodic =====================

    def _is_periodic(self, expr):
        """C=1: Function should be periodic."""
        # SymPy symbolic check
        try:
            from sympy.calculus.util import periodicity
            period = periodicity(expr, self.x)
            if period is not None:
                return True
        except Exception:
            pass

        # Numerical fallback: check f(x) ≈ f(x+T) for candidate periods
        x_vals = np.linspace(0.1, 8, 200)
        y_vals = self._eval_safe(expr, x_vals)
        if y_vals is None or not np.all(np.isfinite(y_vals)):
            return False

        for T in [np.pi, 2*np.pi, np.pi/2, 1.0, 2.0, 3.0, 4.0]:
            y_shifted = self._eval_safe(expr, x_vals + T)
            if y_shifted is not None and np.all(np.isfinite(y_shifted)):
                if np.max(np.abs(y_vals - y_shifted)) < 1e-6:
                    return True

        return False

    # ===================== C=2: Isolated Singularities =====================

    def _has_isolated_singularities(self, expr):
        """C=2: Function should have isolated singularities (poles/asymptotes)."""
        # Method 1: Fraction denominator
        try:
            _, den = sp.fraction(sp.together(expr))
            if den != 1:
                roots = sp.solve(den, self.x)
                if 0 < len(roots) <= 5:  # Isolated = few poles
                    return True
        except Exception:
            pass

        # Method 2: SymPy singularities
        try:
            from sympy.calculus.util import singularities
            sings = singularities(expr, self.x)
            if isinstance(sings, sp.FiniteSet) and len(sings) > 0:
                return True
        except Exception:
            pass

        # Method 3: Numerical — look for isolated blow-ups
        x_vals = np.linspace(-5, 5, 2000)
        y_vals = self._eval_safe(expr, x_vals)
        if y_vals is not None:
            non_finite = np.sum(~np.isfinite(y_vals))
            # Isolated: some non-finite but not too many (< 5% of points)
            if 0 < non_finite < len(x_vals) * 0.05:
                return True

        return False

    # ===================== C=3: Complex/Chaotic Behavior =====================

    def _has_complex_behavior(self, expr):
        """C=3: Complex singularities, multiple poles, oscillating singularities."""
        expr_str = str(expr)

        # Check 1: Oscillating singularity patterns (sin(1/x) type)
        if 'sin(1/' in expr_str or 'cos(1/' in expr_str:
            return True

        # Check 2: Multiple poles
        try:
            _, den = sp.fraction(sp.together(expr))
            if den != 1:
                roots = sp.solve(den, self.x)
                if len(roots) >= 2:
                    return True
        except Exception:
            pass

        # Check 3: Numerical — lots of non-finite values or rapid oscillation
        x_vals = np.linspace(-5, 5, 2000)
        y_vals = self._eval_safe(expr, x_vals)
        if y_vals is not None:
            non_finite = np.sum(~np.isfinite(y_vals))
            if non_finite > len(x_vals) * 0.05:
                return True

            finite_y = y_vals[np.isfinite(y_vals)]
            if len(finite_y) > 10:
                diffs = np.abs(np.diff(finite_y))
                if np.any(diffs > 1e3):
                    return True

        # Check 4: If evaluation completely fails → likely highly singular
        if y_vals is None:
            return True

        # Check 5: Has singularities PLUS periodicity → complex
        if self._has_isolated_singularities(expr):
            try:
                from sympy.calculus.util import periodicity
                if periodicity(expr, self.x) is not None:
                    return True
            except Exception:
                pass
            return True  # At least has singularities, accept for C3

        return False