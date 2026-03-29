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
OUTPUT_FILE = "data/benchmark_tasks.jsonl"
FAILED_FILE = "data/hanging_functions.jsonl"
MANUAL_FILE = "data/manual_formulas.json"
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
                
                manual_list = manual_formulas.get(class_key, [])
                collected_in_class = 0
                
                # --- Обробка (Manual + Auto) в одній черзі ---
                # Спочатку ручні, потім авто
                actual_target = max(target_count, len(manual_list))
                formula_source = manual_list + ["AUTO"] * (target_count + 10) # із запасом
                
                for item in formula_source:
                    if collected_in_class >= actual_target:
                        break
                        
                    try:
                        if item != "AUTO":
                            # Якщо це вже готове завдання зі словника (manual_formulas.json)
                            if isinstance(item, dict) and 'ground_truth' in item:
                                expr_str = item["ground_truth"]["formula"].replace(" ", "")
                                if expr_str in seen_expressions:
                                    collected_in_class += 1
                                    continue
                                
                                append_to_file(OUTPUT_FILE, item)
                                seen_expressions.add(expr_str)
                                collected_in_class += 1
                                total_new += 1
                                continue # Пропускаємо етап валідації для мануальних

                            # Legacy dictionary fallback (just in case)
                            elif isinstance(item, dict):
                                expr = sp.parse_expr(item["formula"])
                                custom_points = item.get("x_points")
                            else:
                                expr = sp.parse_expr(item)
                            is_manual = True
                        else:
                            success, expr, err = DatasetSampler.generate_safe(a, b, c, timeout=15)
                            if not success:
                                logger.warning(f"TIMEOUT Generation: {err}")
                                continue
                            is_manual = False
                    except Exception as e:
                        logger.warning(f"Помилка генерації/парсингу: {e}")
                        continue

                    # Нормалізація рядка
                    if is_manual:
                        expr = sp.simplify(expr)
                    expr_str = str(expr)
                    
                    # ДЕДУПЛІКАЦІЯ
                    if expr_str in seen_expressions or expr_str in failed_expressions:
                        if is_manual: 
                             # Якщо ручна формула вже є, рахуємо її як зроблену
                             collected_in_class += 1 
                        continue

                    # ВАЛІДАЦІЯ (Тільки для авто)
                    if not is_manual:
                        val_success, is_valid, val_err = DatasetSampler.validate_safe(a, b, c, expr, timeout=10)
                        if not val_success or not is_valid:
                            if not val_success:
                                logger.warning(f"TIMEOUT Validation: {expr_str} - {val_err}")
                            continue

                    # --- БЕЗПЕЧНА ОБРОБКА (TIMEOUTS) ---
                    
                    # 1. Метадані
                    meta_success, metadata, meta_err = DatasetSampler.calculate_metadata_safe(expr, timeout=5)
                    
                    if not meta_success:
                        logger.warning(f"TIMEOUT Metadata: {expr_str}")
                        failed_task = TaskExporter.create_task(expr, None, None, config, {"error": meta_err, "stage": "metadata"})
                        append_to_file(FAILED_FILE, failed_task)
                        failed_expressions.add(expr_str)
                        continue

                    # 2. Точки (Тільки для Авто генерації)
                    points_success, (x_vals, y_vals), points_err = DatasetSampler.calculate_points_safe(expr, n_points=25, timeout=3, custom_x=None)
                    
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
                    task["source"] = "manual" if is_manual else "auto" # Додаємо джерело для аналізу
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