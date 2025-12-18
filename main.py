import time
import sympy as sp
import numpy as np
from src.config import ComplexityConfig
from src.generator import ExpressionGenerator
from src.validator import TopologyFilter
from src.sampler import DatasetSampler, TaskExporter
from src.utils import setup_logging, load_manual_formulas, load_seen_expressions, append_to_file
import multiprocessing

# --- CONFIGURATION ---
OUTPUT_FILE = "benchmark_tasks.jsonl"
FAILED_FILE = "hanging_functions.jsonl"
MANUAL_FILE = "manual_formulas.json"
PLAN = np.full((4, 4, 4), 2) 
PLAN[0, 0, 0] = 5

def generate_benchmark_suite():
    logger = setup_logging()
    logger.info(f"=== Початок генерації ===")
    
    # 1. Завантаження контексту
    manual_formulas = load_manual_formulas(MANUAL_FILE)
    seen_expressions = load_seen_expressions(OUTPUT_FILE)
    failed_expressions = load_seen_expressions(FAILED_FILE) # Щоб не "зависати" на тих самих функціях знову
    
    logger.info(f"Завантажено {len(seen_expressions)} існуючих завдань.")
    logger.info(f"Завантажено {len(failed_expressions)} раніше невдалих функцій.")

    total_new = 0

    for a in range(4):
        for b in range(4):
            for c in range(4):
                class_key = f"{a},{b},{c}"
                target_count = PLAN[a, b, c]
                
                # Підрахунок вже наявних завдань цього класу можна зробити точнішим, 
                # але для спрощення покладаємось на загальний seen_expressions
                # (для повної точності треба було б парсити файл і рахувати по класах, 
                # але ми будемо просто намагатися додати нові, доки не досягнемо target в цьому запуску 
                # або можна просто ігнорувати вже наявні і додавати зверху)
                
                logger.info(f"Клас <{class_key}>...")
                
                config = ComplexityConfig(a, b, c)
                gen = ExpressionGenerator(config)
                validator = TopologyFilter(config)
                
                manual_list = manual_formulas.get(class_key, [])
                collected_in_class = 0
                
                # --- Обробка (Manual + Auto) в одній черзі ---
                # Спочатку ручні, потім авто
                formula_source = manual_list + ["AUTO"] * (target_count + 10) # із запасом
                
                for item in formula_source:
                    if collected_in_class >= target_count:
                        break
                        
                    # Отримання виразу
                    try:
                        if item != "AUTO":
                            expr = sp.parse_expr(item)
                            is_manual = True
                        else:
                            expr = gen.generate()
                            is_manual = False
                    except Exception as e:
                        logger.warning(f"Помилка генерації/парсингу: {e}")
                        continue

                    expr_str = str(sp.simplify(expr)) # Нормалізація рядка
                    
                    # ДЕДУПЛІКАЦІЯ
                    if expr_str in seen_expressions or expr_str in failed_expressions:
                        if is_manual: 
                             # Якщо ручна формула вже є, рахуємо її як зроблену
                             collected_in_class += 1 
                        continue

                    # ВАЛІДАЦІЯ (Тільки для авто)
                    if not is_manual and not validator.check(expr):
                        continue

                    # --- БЕЗПЕЧНА ОБРОБКА (TIMEOUTS) ---
                    
                    # 1. Метадані
                    meta_success, metadata, meta_err = DatasetSampler.calculate_metadata_safe(expr, timeout=5)
                    
                    if not meta_success:
                        logger.warning(f"TIMEOUT Metadata: {expr_str}")
                        failed_task = TaskExporter.create_task(expr, None, None, config, {"error": meta_err, "stage": "metadata"})
                        append_to_file(FAILED_FILE, failed_task)
                        failed_expressions.add(expr_str)
                        continue # Переходимо до наступного, бо метадані критичні (залежить від ваших вимог)

                    # 2. Точки
                    points_success, (x_vals, y_vals), points_err = DatasetSampler.calculate_points_safe(expr, timeout=3)
                    
                    if not points_success:
                        logger.warning(f"TIMEOUT Points: {expr_str}")
                        # Зберігаємо те, що встигли (метадані)
                        metadata["error"] = points_err
                        failed_task = TaskExporter.create_task(expr, None, None, config, metadata)
                        append_to_file(FAILED_FILE, failed_task)
                        failed_expressions.add(expr_str)
                        continue

                    # Успіх
                    task = TaskExporter.create_task(expr, x_vals, y_vals, config, metadata)
                    append_to_file(OUTPUT_FILE, task)
                    seen_expressions.add(expr_str)
                    collected_in_class += 1
                    total_new += 1
                    
                    if total_new % 10 == 0:
                        logger.info(f"Згенеровано {total_new} нових завдань...")

if __name__ == "__main__":
    # Необхідно для multiprocessing на Windows
    multiprocessing.freeze_support() 
    generate_benchmark_suite()