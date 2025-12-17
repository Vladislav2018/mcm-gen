import random
import sympy as sp
from .config import ComplexityConfig

class ExpressionGenerator:
    """Генерація виразів згідно з Axis A та Axis B[cite: 124]."""
    def __init__(self, config: ComplexityConfig):
        self.config = config

    def _generate_recursive(self, depth: int) -> sp.Expr:
        if depth >= self.config.max_depth or (depth > 0 and random.random() < 0.2):
            return self.config.x if random.random() < 0.7 else sp.Integer(random.randint(1, 5))

        op = random.choice(self.config.available_ops)
        if op in [sp.Add, sp.Mul]:
            return op(self._generate_recursive(depth + 1), self._generate_recursive(depth + 1))
        elif op == sp.Pow:
            return op(self._generate_recursive(depth + 1), random.randint(2, 3))
        return op(self._generate_recursive(depth + 1))

    def generate(self) -> sp.Expr:
        return sp.simplify(self._generate_recursive(0))