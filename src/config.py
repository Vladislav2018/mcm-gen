import sympy as sp

class ComplexityConfig:
    """Конфігурація MCM: мапінг вектора <A, B, C> на параметри генерації."""
    def __init__(self, a: int, b: int, c: int):
        self.a, self.b, self.c = a, b, c
        self.x = sp.Symbol('x')
        
        # Axis A (Structure): Глибина дерева [cite: 53, 59, 62, 66]
        self.depth_map = {0: 1, 1: 3, 2: 6, 3: 10}
        self.max_depth = self.depth_map.get(a, 3)
        
        # Axis B (Semantics): Набори операторів [cite: 74, 78, 82, 85]
        op_sets = {
            0: [sp.Add, sp.Mul, sp.Pow],
            1: [sp.sin, sp.cos, sp.exp, sp.log],
            2: [sp.Abs, sp.floor, sp.Piecewise],
            3: [sp.besselj, sp.gamma, sp.erf]
        }
        self.available_ops = []
        for i in range(b + 1):
            self.available_ops.extend(op_sets[i])