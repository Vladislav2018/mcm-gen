import sympy as sp

class ComplexityConfig:
    def __init__(self, a: int, b: int, c: int):
        self.a, self.b, self.c = a, b, c
        self.x = sp.Symbol('x')
        # Зменшено глибину для економії ваших ресурсів
        self.max_depth = {0: 1, 1: 2, 2: 3, 3: 5}.get(a, 2)
        
        op_sets = {
            0: [sp.Add, sp.Mul, sp.Pow], # Sub не потрібен, Mul(-1, ...) його замінить
            1: [sp.sin, sp.cos, sp.tan, sp.exp, sp.log],
            2: [sp.Abs, sp.floor, sp.Piecewise, sp.factorial],
            3: [sp.besselj, sp.gamma, sp.erf, sp.zeta]
        }
        self.available_ops = []
        for i in range(b + 1):
            self.available_ops.extend(op_sets[i])