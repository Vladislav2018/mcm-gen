import sympy as sp

# Винесено на рівень модуля для доступу з інших модулів
OP_SETS = {
    0: {sp.Add, sp.Mul, sp.Pow}, # Базові
    1: {sp.sin, sp.cos, sp.tan, sp.exp, sp.log},
    2: {sp.Abs, sp.floor, sp.Piecewise, sp.factorial},
    3: {sp.besselj, sp.gamma, sp.erf, sp.zeta}
}

class ComplexityConfig:
    def __init__(self, a: int, b: int, c: int):
        self.a, self.b, self.c = a, b, c
        # ВАЖЛИВО: real=True допомагає уникнути появи re(), im(), atan2() при спрощеннях
        self.x = sp.Symbol('x', real=True)
        
        self.max_depth = {0: 1, 1: 2, 2: 3, 3: 5}.get(a, 2)
        
        self.available_ops = []
        for i in range(b + 1):
            self.available_ops.extend(list(OP_SETS[i]))