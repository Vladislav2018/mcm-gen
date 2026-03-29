"""MCM-Gen Expression Generator v2.0

Template-based generation with parametric diversity.
Generates expressions matching target complexity vector (A, B, C):
  A: Structural depth (tree depth)
  B: Operator set level (0=arith, 1=trig/exp, 2=non-smooth, 3=special)
  C: Topological behavior (0=regular, 1=periodic, 2=singular, 3=chaotic)
"""

import random
import sympy as sp
from .config import ComplexityConfig, OP_SETS


class ExpressionGenerator:
    """Generates mathematical expressions matching a target complexity vector."""

    def __init__(self, config: ComplexityConfig):
        self.config = config
        self.x = config.x

    # ===================== Random Parameter Helpers =====================

    def _coeff(self):
        """Random non-zero integer in [-5, 5]."""
        return sp.Integer(random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]))

    def _small_coeff(self):
        """Random non-zero integer in [-3, 3]."""
        return sp.Integer(random.choice([-3, -2, -1, 1, 2, 3]))

    def _pos(self):
        """Random positive integer in [1, 5]."""
        return sp.Integer(random.randint(1, 5))

    def _shift(self):
        """Random shift for singularity positions (avoids 0)."""
        vals = [sp.Rational(-3), sp.Rational(-2), sp.Rational(-3, 2),
                sp.Rational(-1), sp.Rational(-1, 2),
                sp.Rational(1, 2), sp.Rational(1), sp.Rational(3, 2),
                sp.Rational(2), sp.Rational(3)]
        return random.choice(vals)

    def _pos_val(self):
        """Random positive value for denominators/offsets."""
        return random.choice([sp.Rational(1, 2), sp.Integer(1), sp.Rational(3, 2),
                              sp.Integer(2), sp.Integer(3)])

    def _freq(self):
        """Random frequency for periodic functions."""
        return random.choice([sp.Integer(1), sp.Integer(2), sp.Integer(3),
                              sp.Integer(4), sp.Integer(5)])

    # ===================== Inner Expression Builders (by B level) =====================

    def _build_inner(self):
        """Build inner expression respecting A (depth) and B (operators)."""
        depth = self.config.max_depth
        b = self.config.b

        if b == 0:
            return self._inner_b0(depth)
        elif b == 1:
            return self._inner_b1(depth)
        elif b == 2:
            return self._inner_b2(depth)
        else:
            return self._inner_b3(depth)

    def _inner_b0(self, depth):
        """B=0: Polynomial/rational using only +, -, *, ^."""
        x = self.x
        if depth <= 1:
            return random.choice([
                self._coeff() * x + self._coeff(),
                x ** random.choice([2, 3]),
                self._coeff() * x ** 2,
                self._coeff() * x,
            ])
        if depth == 2:
            return random.choice([
                self._coeff() * x**2 + self._coeff() * x + self._coeff(),
                (self._coeff() * x + self._coeff()) ** 2,
                self._coeff() * x**3 + self._coeff() * x,
                (self._small_coeff() * x + self._coeff()) * (x + self._coeff()),
            ])
        if depth == 3:
            return random.choice([
                (self._coeff() * x**2 + self._coeff()) ** 2,
                self._coeff() * x**4 + self._coeff() * x**2 + self._coeff(),
                self._inner_b0(2) * self._inner_b0(1),
                self._inner_b0(2) + self._coeff() * self._inner_b0(1),
            ])
        # depth >= 4
        return random.choice([
            self._inner_b0(3) + self._coeff() * self._inner_b0(2),
            (self._inner_b0(2)) ** 2 + self._coeff() * x,
            self._inner_b0(3) * self._inner_b0(1),
        ])

    def _inner_b1(self, depth):
        """B=1: Expressions with trig/exp/log (must contain at least one)."""
        x = self.x
        sub = self._inner_b0(max(1, depth - 1))

        if depth <= 1:
            return random.choice([
                sp.sin(sub), sp.cos(sub),
                sp.exp(sub / self._pos()),
                sp.log(sub**2 + self._pos_val()),
            ])
        if depth == 2:
            f = self._freq()
            return random.choice([
                sp.sin(f * x + self._coeff()),
                sp.cos(f * x) + self._coeff() * sp.sin(f * x),
                sp.exp(-x**2 / self._pos_val()),
                sp.log(x**2 + self._pos_val()) * self._coeff(),
                sp.exp(self._small_coeff() * x),
                self._coeff() * sp.sin(f * x) * sp.cos(f * x),
            ])
        if depth == 3:
            return random.choice([
                sp.sin(self._inner_b0(2)),
                sp.cos(self._inner_b0(2)),
                sp.exp(-self._inner_b0(1)**2 / self._pos_val()),
                sp.sin(self._freq() * x) * sp.exp(-x**2 / self._pos_val()),
                sp.log(self._inner_b0(2)**2 + self._pos_val()),
                self._inner_b0(2) * sp.sin(self._freq() * x),
                sp.exp(sp.sin(self._freq() * x)),
            ])
        # depth >= 4
        return random.choice([
            sp.sin(sp.cos(self._inner_b0(2))),
            sp.sin(sp.exp(self._inner_b0(1) / self._pos())),
            sp.sin(self._inner_b0(2)**2 + sp.cos(self._inner_b0(1))),
            self._inner_b0(2) * sp.exp(sp.sin(self._freq() * x)),
            sp.log(sp.sin(self._freq() * x)**2 + self._pos_val()),
            sp.sin(self._inner_b1(2)),
        ])

    def _inner_b2(self, depth):
        """B=2: Non-smooth operators (Abs/floor/sign/Piecewise). Must contain B=2 op."""
        if random.random() < 0.5 and depth >= 2:
            base = self._inner_b1(max(1, depth - 1))
        else:
            base = self._inner_b0(max(1, depth - 1))

        wrapped = random.choice([
            sp.Abs(base),
            sp.floor(base),
            sp.sign(base),
            sp.Piecewise((base, self.x > self._shift()), (-base, True)),
            sp.Piecewise((base, self.x > 0), (self._coeff() * self.x, True)),
        ])

        if depth >= 3 and random.random() < 0.5:
            wrapped = wrapped + self._coeff() * self._inner_b0(1)

        return wrapped

    def _inner_b3(self, depth):
        """B=3: Special functions (gamma/bessel/erf). Must contain B=3 op."""
        sub = self._inner_b0(max(1, depth - 1))

        wrapped = random.choice([
            sp.gamma(sp.Abs(sub) + sp.Rational(11, 10)),
            sp.erf(sub),
            sp.besselj(random.choice([0, 1]), sp.Abs(sub)),
            sp.gamma(sub**2 + sp.Rational(11, 10)),
            sp.besselj(0, sub**2 + 1),
            sp.erf(sub**2),
        ])

        if depth >= 3 and random.random() < 0.5:
            extra = random.choice([sp.sin, sp.cos])(self._inner_b0(1))
            wrapped = wrapped * extra if random.random() < 0.5 else wrapped + self._coeff() * extra

        return wrapped

    # ===================== Topology Wrappers (by C level) =====================

    def _wrap_c0(self, inner):
        """C=0: Regular, smooth function. No singularities, no periodicity."""
        return inner

    def _wrap_c1(self, inner):
        """C=1: Periodic function. Must have detectable period."""
        f = self._freq()
        x = self.x

        # For B=0: polynomial cannot be periodic, so we add sin/cos
        # (acknowledged as theoretical extension in the paper)
        if self.config.b == 0:
            return random.choice([
                sp.sin(f * x + self._coeff()),
                sp.cos(f * x) + self._coeff(),
                sp.sin(f * x) * sp.cos(f * x),
            ])

        # For B >= 1: wrap inner or create periodic
        return random.choice([
            sp.sin(inner), sp.cos(inner),
            sp.sin(f * x + self._coeff()),
            sp.cos(f * x) + self._coeff() * sp.sin(f * x),
            sp.sin(f * x) * inner,
        ])

    def _wrap_c2(self, inner):
        """C=2: Isolated singularities (poles, vertical asymptotes)."""
        s = self._shift()
        return random.choice([
            inner / (self.x - s),
            self._coeff() / (self.x - s),
            (self._coeff() * self.x + self._coeff()) / (self.x - s),
            inner / (self.x - s) + self._coeff(),
            self._coeff() * self.x**2 / (self.x - s),
        ])

    def _wrap_c3(self, inner):
        """C=3: Complex singularities, multiple poles, oscillating behavior."""
        s1 = self._shift()
        s2 = self._shift()
        while abs(float(s2 - s1)) < 0.3:
            s2 = self._shift()

        return random.choice([
            inner * sp.sin(1 / (self.x - s1)),
            inner / ((self.x - s1) * (self.x - s2)),
            inner / (self.x - s1) + self._coeff() / (self.x - s2),
            sp.sin(1 / (self.x - s1)) + sp.cos(1 / (self.x - s2)),
            inner * sp.sin(1 / (self.x - s1)) / (self.x - s2),
        ])

    # ===================== Verification =====================

    def _verify_complexity(self, expr):
        """Verify expression matches target complexity vector."""
        if not expr.has(self.x):
            return False
        if expr.has(sp.oo, sp.zoo, sp.nan):
            return False

        # No re, im, atan2
        for atom in expr.atoms():
            if isinstance(atom, (sp.re, sp.im)) or str(atom) == 'atan2':
                return False

        # No huge numbers
        for num in expr.atoms(sp.Number):
            if abs(num) > 50:
                return False

        # B-level check: must use operators from target B level
        if self.config.b >= 1:
            expr_str = str(expr)
            checks = {
                1: ['sin', 'cos', 'tan', 'exp', 'log'],
                2: ['Abs', 'floor', 'sign', 'Piecewise', 'factorial'],
                3: ['gamma', 'besselj', 'erf', 'zeta'],
            }
            found = any(name in expr_str for name in checks.get(self.config.b, []))
            if not found:
                return False

        return True

    # ===================== Main Entry Point =====================

    def generate(self):
        """Generate a single expression matching the complexity vector.

        Returns:
            sp.Expr or None if generation failed.
        """
        for _ in range(100):
            try:
                inner = self._build_inner()
                expr = self._wrap_topology(inner)

                simplified = sp.simplify(expr)

                if self._verify_complexity(simplified):
                    return simplified
            except Exception:
                continue

        return None

    def _wrap_topology(self, inner):
        """Apply topology wrapper based on C level."""
        c = self.config.c
        if c == 0: return self._wrap_c0(inner)
        elif c == 1: return self._wrap_c1(inner)
        elif c == 2: return self._wrap_c2(inner)
        else: return self._wrap_c3(inner)