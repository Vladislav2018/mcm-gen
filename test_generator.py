"""Quick test of the new generator v2."""
import sympy as sp
from src.config import ComplexityConfig
from src.generator import ExpressionGenerator

print("=== Generator v2 Test ===\n")
successes = 0
failures = 0

for a in range(4):
    for b in range(4):
        for c in range(4):
            config = ComplexityConfig(a, b, c)
            gen = ExpressionGenerator(config)
            expr = gen.generate()
            if expr is not None:
                successes += 1
                if (a + b + c) % 8 == 0:  # Print a sample every 8 classes
                    print(f"  ({a},{b},{c}): {expr}")
            else:
                failures += 1
                print(f"  ({a},{b},{c}): FAILED")

print(f"\nResults: {successes}/64 success, {failures}/64 failed")
