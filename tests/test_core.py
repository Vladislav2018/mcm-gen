import pytest
import sympy as sp
from src.config import ComplexityConfig
from src.generator import ExpressionGenerator
from src.sampler import DatasetSampler

def test_expression_contains_x():
    """Перевірка, що генератор завжди видає функцію від X."""
    config = ComplexityConfig(1, 1, 0)
    gen = ExpressionGenerator(config)
    expr = gen.generate()
    assert expr.has(sp.Symbol('x')), f"Вираз {expr} не містить змінну x"

def test_sampler_output_format():
    """Перевірка, що семплер видає рівно n точок і вони є дійсними."""
    x = sp.Symbol('x')
    expr = x**2 + 1
    n = 25
    x_vals, y_vals, meta = DatasetSampler.get_data(expr, n_points=n)
    
    assert len(x_vals) == n
    assert len(y_vals) == n
    assert "domain" in meta

@pytest.mark.parametrize("a,b,c", [(0,0,0), (3,3,3)])
def test_config_mapping(a, b, c):
    """Перевірка, що конфіг правильно мапить рівні на глибину."""
    config = ComplexityConfig(a, b, c)
    assert config.max_depth > 0