import json
import os
import time
import logging
import numpy as np
import sympy as sp
from src.config import ComplexityConfig
from src.generator import ExpressionGenerator
from src.validator import TopologyFilter
from src.sampler import DatasetSampler, TaskExporter
from src.utils import setup_logging

# Ініціалізація професійного логування
logger = setup_logging()

# --- 1. МАТРИЦЯ ПЛАНУВАННЯ (A, B, C) ---
PLAN = np.full((4, 4, 4), 2)  # За замовчуванням по 2 завдання на клас
PLAN[0, 0, 0] = 5  # Приклад: більше простих завдань

MANUAL_OVERRIDE = {
    # Класи з асимптотами (C=2), де випадкова генерація часто хибить
    "0,0,2": ["1/(x-1)", "x/(x+2)", "1/(x**2 - 4)"],
    "0,1,2": ["tan(x)", "exp(1/x)", "1/cos(x)"],
    "0,2,2": ["1/abs(x-0.5)"],
    "0,3,2": ["1/erf(x)", "gamma(x)/x"],
    
    "1,0,2": ["(x+1)/(x-1.2)", "(x**2)/(x+0.5)"],
    "1,1,2": ["1/sin(x)", "exp(x)/(x-2)"],
    "1,2,2": ["floor(x)/(x-1)", "abs(x)/(x+1.5)"],
    "1,3,2": ["zeta(x+1.1)", "1/gamma(x)"], # Zeta має полюс в 1
    
    "2,0,2": ["(x**2+1)/(x*(x-2))", "1/(x**3-x)"],
    "2,1,2": ["sin(x)/(cos(x)-0.5)", "log(abs(x-1))"],
    "2,2,2": ["abs(x)/(x-1)", "factorial(floor(abs(x)))/(x+2)"],
    "2,3,2": ["besselj(0, x)/(x-0.5)", "erf(x)/(x**2-1)"],
    
    "3,0,2": ["(x**3+x)/(x**2-4)", "x**5/(x-0.1)"],
    "3,1,2": ["log(abs(sin(x)))", "exp(tan(x))"],
    "3,2,2": ["Piecewise((1/x, x>0.1), (0, True))", "floor(exp(x))/(x-2.5)"],
    "3,3,2": ["gamma(x)/(x-2.5)", "1/zeta(abs(x)+1.1)"]
}

def generate_benchmark_suite(filename="benchmark_tasks.jsonl"):
    """Основна функція генерації бенчмарку з логуванням та контролем унікальності."""
    logger.info(f"=== Початок генерації бенчмарку: {filename} ===")
    start_time = time.time()
    
    if os.path.exists(filename):
        os.remove(filename)
        logger.debug(f"Старий файл {filename} видалено.")

    seen_expressions = set()
    total_collected = 0

    with open(filename, "a", encoding="utf-8") as f:
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    class_key = f"{a},{b},{c}"
                    target_count = PLAN[a, b, c]
                    manual_list = MANUAL_OVERRIDE.get(class_key, [])
                    
                    logger.info(f"Обробка класу <{class_key}>: План={target_count}, Ручних={len(manual_list)}")
                    
                    config = ComplexityConfig(a, b, c)
                    gen = ExpressionGenerator(config)
                    validator = TopologyFilter(config)
                    
                    collected_in_class = 0
                    
                    # 1. Обробка ручних формул (Manual Override)
                    for formula_str in manual_list:
                        try:
                            expr = sp.parse_expr(formula_str)
                            x_vals, y_vals, meta = DatasetSampler.get_data(expr)
                            if x_vals is not None:
                                task = TaskExporter.create_task(expr, x_vals, y_vals, config, meta)
                                f.write(json.dumps(task, ensure_ascii=False) + "\n")
                                seen_expressions.add(str(sp.simplify(expr)))
                                collected_in_class += 1
                                logger.debug(f"  [Manual] Додано: {formula_str}")
                        except Exception as e:
                            logger.error(f"  [Manual] Помилка у формулі '{formula_str}': {e}")

                    # 2. Автоматична процедурна генерація
                    attempts = 0
                    max_attempts = 40
                    while collected_in_class < target_count and attempts < max_attempts:
                        attempts += 1
                        try:
                            expr = gen.generate()
                            expr_str = str(expr)
                            
                            if expr_str in seen_expressions:
                                continue

                            if validator.check(expr):
                                x_vals, y_vals, meta = DatasetSampler.get_data(expr)
                                if x_vals is not None:
                                    task = TaskExporter.create_task(expr, x_vals, y_vals, config, meta)
                                    f.write(json.dumps(task, ensure_ascii=False) + "\n")
                                    seen_expressions.add(expr_str)
                                    collected_in_class += 1
                                    logger.debug(f"  [Auto] Додано ({attempts} спроба): {expr_str}")
                        except Exception as e:
                            logger.warning(f"  [Auto] Збій під час генерації спроби {attempts}: {e}")
                    
                    total_collected += collected_in_class
                    logger.info(f"Клас <{class_key}> завершено. Разом: {collected_in_class}")

    duration = time.time() - start_time
    logger.info(f"=== Генерацію завершено за {duration:.2f} сек. Всього завдань: {total_collected} ===")

def run_with_profiling():
    """Запуск генерації з профілюванням продуктивности."""
    import cProfile
    import pstats
    
    logger.info("Запуск профілювання продуктивності...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    generate_benchmark_suite(filename="profile_test.jsonl")
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20) # Вивід топ-20 найповільніших операцій

if __name__ == "__main__":
    # Щоб запустити профілювання, змініть на run_with_profiling()
    generate_benchmark_suite()