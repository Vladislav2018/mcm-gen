import numpy as np
import sympy as sp
import random
import time
from typing import List, Optional, Tuple, Dict, Any

class ComplexityConfig:
    """
    Клас конфігурації Багатовимірної матриці складності (MCM).
    Відповідає за мапінг рівнів <A, B, C> на параметри генерації.
    """
    def __init__(self, a: int, b: int, c: int):
        self.a = a  # Axis A: Structural
        self.b = b  # Axis B: Semantic
        self.c = c  # Axis C: Topological
        
        # Налаштування Axis A (Structure)
        self.depth_map = {0: 1, 1: 3, 2: 6, 3: 10}
        self.max_depth = self.depth_map.get(a, 3)
        
        # Налаштування Axis B (Semantics) - Оператори SymPy
        self.x = sp.Symbol('x')
        self.op_sets = {
            0: [sp.Add, sp.Mul, sp.Pow], # Арифметичний базис
            1: [sp.sin, sp.cos, sp.exp, sp.log], # Трансцендентні
            2: [sp.Abs, sp.floor, sp.Piecewise], # Логічні/Недиференційовні
            3: [sp.besselj, sp.gamma, sp.erf] # Спеціальні функції
        }
        
        # Кумулятивний набір операторів згідно з рівнем B
        self.available_ops = []
        for i in range(b + 1):
            self.available_ops.extend(self.op_sets[i])

class ExpressionGenerator:
    """
    Модуль синтаксичного синтезу (Axes A & B).
    Використовує метод стохастичного зростання дерев.
    """
    def __init__(self, config: ComplexityConfig):
        self.config = config

    def _generate_recursive(self, current_depth: int) -> sp.Expr:
        # Базовий випадок: термінали (x або константи)
        if current_depth >= self.config.max_depth or (current_depth > 0 and random.random() < 0.2):
            return self.config.x if random.random() < 0.7 else sp.Integer(random.randint(1, 5))

        op = random.choice(self.config.available_ops)
        
        # Обробка арності операторів
        if op in [sp.Add, sp.Mul]:
            return op(self._generate_recursive(current_depth + 1), 
                      self._generate_recursive(current_depth + 1))
        elif op == sp.Pow:
            return op(self._generate_recursive(current_depth + 1), random.randint(2, 3))
        elif op == sp.Piecewise:
            # Спрощене розгалуження для B2
            expr = self._generate_recursive(current_depth + 1)
            return sp.Piecewise((expr, self.config.x > 0), (-expr, True))
        else:
            # Унарні функції (sin, exp, besselj тощо)
            return op(self._generate_recursive(current_depth + 1))

    def generate(self) -> sp.Expr:
        """Генерує вираз та проводить базове спрощення."""
        expr = self._generate_recursive(0)
        return sp.simplify(expr)

class TopologyFilter:
    """
    Модуль топологічної валідації (Axis C).
    Перевіряє відповідність виразу якісній поведінці.
    """
    def __init__(self, config: ComplexityConfig):
        self.config = config
        self.x = config.x

    def check_topology(self, expr: sp.Expr) -> bool:
        level = self.config.c
        try:
            if level == 0: return self._is_regular(expr)
            if level == 1: return self._is_periodic_or_symmetric(expr)
            if level == 2: return self._has_asymptotes(expr)
            if level == 3: return self._is_singular_or_chaotic(expr)
        except:
            return False
        return False

    def _is_regular(self, expr: sp.Expr) -> bool:
        # C0: Відсутність розривів та скінченність у діапазоні
        test_vals = np.linspace(-5, 5, 100)
        f = sp.lambdify(self.x, expr, 'numpy')
        res = f(test_vals)
        return np.all(np.isfinite(res)) and not sp.degree(expr, self.x) == 0

    def _is_periodic_or_symmetric(self, expr: sp.Expr) -> bool:
        # C1: Перевірка періодичності через SymPy або парності
        is_even = (expr - expr.subs(self.x, -self.x)).simplify() == 0
        has_trig = any(isinstance(node, (sp.sin, sp.cos)) for node in sp.preorder_traversal(expr))
        return is_even or has_trig

    def _has_asymptotes(self, expr: sp.Expr) -> bool:
        # C2: Пошук полюсів (знаменник стає 0)
        num, den = sp.fraction(expr)
        return den != 1 and len(sp.solve(den, self.x)) > 0

    def _is_singular_or_chaotic(self, expr: sp.Expr) -> bool:
        # C3: Складні осциляції або Piecewise
        has_complex_nodes = any(isinstance(node, (sp.Piecewise, sp.Abs)) for node in sp.preorder_traversal(expr))
        # Перевірка на швидкі осциляції типу sin(1/x)
        return has_complex_nodes or ("1/x" in str(expr))

class DatasetGenerator:
    """Генератор безшумних даних (Етап 3)."""
    @staticmethod
    def sample(expr: sp.Expr, x_range: Tuple[float, float], n_points: int):
        x_sym = list(expr.free_symbols)[0] if expr.free_symbols else sp.Symbol('x')
        f_np = sp.lambdify(x_sym, expr, 'numpy')
        
        x_vals = np.linspace(x_range[0], x_range[1], n_points)
        try:
            y_vals = f_np(x_vals)
            # Очищення від комплексних чисел, які можуть виникнути
            if np.iscomplexobj(y_vals):
                y_vals = np.real(y_vals)
            return x_vals, y_vals
        except Exception as e:
            return None, None

class MCMGenerator:
    """Оркестратор: реалізує стратегію Rejection Sampling."""
    def __init__(self, a, b, c):
        self.config = ComplexityConfig(a, b, c)
        self.gen = ExpressionGenerator(self.config)
        self.validator = TopologyFilter(self.config)

    def produce(self, timeout=10) -> Tuple[sp.Expr, np.ndarray, np.ndarray]:
        start_time = time.time()
        while time.time() - start_time < timeout:
            expr = self.gen.generate()
            if self.validator.check_topology(expr):
                x_v, y_v = DatasetGenerator.sample(expr, (-5, 5), 100)
                if x_v is not None and np.all(np.isfinite(y_v)):
                    return expr, x_v, y_v
        raise TimeoutError(f"Не вдалося згенерувати вираз для класу <{self.config.a},{self.config.b},{self.config.c}>")

# --- ПРИКЛАД ВИКОРИСТАННЯ ---
if __name__ == "__main__":
    print("--- MCM-Gen: Процедурна генерація розпочата ---")
    
    test_cases = [
        (0, 0, 0), # Baseline: Trivial Arithmetic
        (2, 1, 1), # High Structure + Transcendental + Periodic
        (3, 3, 2), # Recursive + Special Functions + Asymptotic
    ]

    for a, b, c in test_cases:
        print(f"\nГенерація класу <A:{a}, B:{b}, C:{c}>...")
        mcm = MCMGenerator(a, b, c)
        try:
            expr, x, y = mcm.produce()
            print(f"Успіх! Формула: {expr}")
            print(f"Перші 5 точок Y: {y[:5]}")
        except Exception as e:
            print(f"Помилка: {e}")