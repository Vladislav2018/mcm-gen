import random
import sympy as sp
from .config import ComplexityConfig, OP_SETS

class ExpressionGenerator:
    def __init__(self, config: ComplexityConfig):
        self.config = config

    def _generate_recursive(self, depth: int) -> sp.Expr:
        if depth >= self.config.max_depth:
            # Генеруємо лише малі числа, щоб уникнути гігантських коефіцієнтів
            if random.random() < 0.8:
                return self.config.x
            else:
                return sp.Integer(random.randint(1, 5))

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
                # Використовуємо float 0.5 замість Rational(1, 2), щоб уникнути sqrt()
                exp = random.choice([2, 3, 0.5, -1]) 
                return sp.Pow(base, exp)
                
            if op == sp.besselj:
                return op(random.randint(0, 1), self._generate_recursive(depth + 1))

            if op == sp.factorial and depth > 1:
                return self.config.x

            return op(self._generate_recursive(depth + 1))
        except Exception:
            return self.config.x

    def _verify_complexity(self, expr: sp.Expr) -> bool:
        """Перевірка складності та валідності операторів."""
        atoms = expr.atoms()
        
        # 1. Заборона re, im, atan2
        for atom in atoms:
             if isinstance(atom, (sp.re, sp.im)) or str(atom) == 'atan2':
                 return False

        # 2. Фільтр ВЕЛИКИХ ЧИСЕЛ
        # Перевіряємо всі числа у виразі
        for num in expr.atoms(sp.Number):
            if abs(num) > 50: # Якщо є число більше 50 — відкидаємо
                return False

        # 3. Перевірка цільової складності B
        if self.config.b == 0:
            return True

        target_ops = OP_SETS[self.config.b]
        has_target = False
        for atom in expr.atoms():
            if atom.func in target_ops:
                has_target = True
                break
        
        return has_target

    def generate(self) -> sp.Expr:
        for _ in range(50):
            expr = self._generate_recursive(0)
            
            # --- ЗМІНИ ТУТ ---
            # 1. Прибрали sp.nsimplify(expr), який робив sqrt і великі дроби
            # 2. Залишили звичайний simplify(), але обережно
            try:
                # simplify іноді може перетворити x**0.5 у sqrt(x), тому
                # можна спробувати спочатку без нього, або використовувати конкретні стратегії.
                # Але для "чистки" виразу (x+x -> 2x) він потрібен.
                simplified = sp.simplify(expr)
            except:
                continue
            
            # Якщо simplify все ж зробив sqrt (наприклад, через згортання), 
            # можна примусово замінити Rational(1, 2) на Float(0.5)
            # Але зазвичай, якщо на вході були float, simplify їх залишає.
            
            # Фільтри
            if not simplified.has(self.config.x):
                continue
            
            if simplified.has(sp.oo, sp.zoo, sp.nan):
                continue
            
            if not self._verify_complexity(simplified):
                continue

            return simplified
                
        return self.config.x