import random
import sympy as sp
from .config import ComplexityConfig

class ExpressionGenerator:
    def __init__(self, config: ComplexityConfig):
        self.config = config

    def _generate_recursive(self, depth: int) -> sp.Expr:
        if depth >= self.config.max_depth:
            return self.config.x if random.random() < 0.8 else sp.Integer(random.randint(1, 5))

        # Форсування асимптот для Axis C=2
        if self.config.c == 2 and depth == 0:
            denom = (self.config.x - random.randint(-2, 2))
            return self._generate_recursive(depth + 1) / (denom if denom != 0 else 1)

        op = random.choice(self.config.available_ops)
        
        try:
            if op == sp.Piecewise:
                expr = self._generate_recursive(depth + 1)
                cond = self.config.x > random.randint(-3, 3)
                return sp.Piecewise((expr, cond), (random.choice([0, -expr]), True))
            
            if op in [sp.Add, sp.Mul]:
                return op(self._generate_recursive(depth + 1), self._generate_recursive(depth + 1))
            
            if op == sp.Pow:
                base = self._generate_recursive(depth + 1)
                # Уникаємо x^1.0 та x^1
                exp = random.choice([2, 3, 0.5])
                return sp.Pow(base, exp)
                
            if op == sp.besselj:
                return op(random.randint(0, 2), self._generate_recursive(depth + 1))

            # Обмеження на вкладеність факторіалів (щоб не зависало)
            if op == sp.factorial and depth > 1:
                return self.config.x

            return op(self._generate_recursive(depth + 1))
        except Exception:
            return self.config.x

    def generate(self) -> sp.Expr:
        for _ in range(15):  # Збільшено кількість спроб
            expr = self._generate_recursive(0)
            
            # 1. Примусове спрощення та прибирання x^1.0 (float -> int)
            simplified = sp.nsimplify(expr).simplify()
            
            # 2. Фільтр: Чи є в рівнянні X?
            if not simplified.has(self.config.x):
                continue
            
            # 3. Фільтр: Чи не перетворилося все в нескінченність?
            if not simplified.has(sp.oo, sp.zoo, sp.nan) and simplified != 0:
                return simplified
                
        return self.config.x + random.randint(1, 10) # Fallback