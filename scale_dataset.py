"""MCM-Gen Dataset Scaling Script

Generates additional tasks to reach target count per class.
Uses the improved template-based generator.

Usage: python scale_dataset.py [--target N] [--timeout T]
"""

import json
import time
import argparse
import multiprocessing
from collections import Counter
from src.sampler import DatasetSampler, TaskExporter
from src.config import ComplexityConfig
from src.utils import setup_logging, load_seen_expressions, append_to_file

OUTPUT_FILE = "data/benchmark_tasks.jsonl"
FAILED_FILE = "data/hanging_functions.jsonl"


def count_by_class(filepath):
    """Count tasks per complexity class."""
    counts = Counter()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                cv = d['complexity_vector']
                counts[(cv['a'], cv['b'], cv['c'])] += 1
    except FileNotFoundError:
        pass
    return counts


def scale_dataset(target_per_class=15, gen_timeout=20, max_attempts_per_task=30):
    """Scale dataset to reach target tasks per class."""
    logger = setup_logging()
    logger.info(f"=== Масштабування датасету: ціль {target_per_class} задач/клас ===")

    seen = load_seen_expressions(OUTPUT_FILE)
    failed = load_seen_expressions(FAILED_FILE)
    current_counts = count_by_class(OUTPUT_FILE)

    total_needed = 0
    total_generated = 0

    for a in range(4):
        for b in range(4):
            for c in range(4):
                key = (a, b, c)
                current = current_counts.get(key, 0)
                needed = max(0, target_per_class - current)
                total_needed += needed

                if needed == 0:
                    logger.info(f"  Клас ({a},{b},{c}): {current} ≥ {target_per_class} ✓")
                    continue

                logger.info(f"  Клас ({a},{b},{c}): {current} → потрібно ще {needed}")

                generated = 0
                attempts = 0

                while generated < needed and attempts < max_attempts_per_task * needed:
                    attempts += 1

                    # 1. Generate
                    success, expr, err = DatasetSampler.generate_safe(
                        a, b, c, timeout=gen_timeout
                    )
                    if not success:
                        if attempts % 10 == 0:
                            logger.warning(f"    Спроба {attempts}: генерація не вдалась - {err}")
                        continue

                    expr_str = str(expr)

                    # 2. Deduplication
                    if expr_str in seen or expr_str in failed:
                        continue

                    # 3. Validate topology
                    val_success, is_valid, val_err = DatasetSampler.validate_safe(
                        a, b, c, expr, timeout=10
                    )
                    if not val_success or not is_valid:
                        continue

                    # 4. Metadata
                    meta_success, metadata, meta_err = DatasetSampler.calculate_metadata_safe(
                        expr, timeout=5
                    )
                    if not meta_success:
                        logger.warning(f"    TIMEOUT Metadata: {expr_str[:50]}...")
                        failed_task = TaskExporter.create_task(
                            expr, None, None,
                            ComplexityConfig(a, b, c),
                            {"error": meta_err, "stage": "metadata"}
                        )
                        append_to_file(FAILED_FILE, failed_task)
                        failed.add(expr_str)
                        continue

                    # 5. Sample points
                    pts_success, (x_vals, y_vals), pts_err = DatasetSampler.calculate_points_safe(
                        expr, n_points=25, timeout=3
                    )
                    if not pts_success:
                        logger.warning(f"    TIMEOUT Points: {expr_str[:50]}...")
                        metadata["error"] = pts_err
                        failed_task = TaskExporter.create_task(
                            expr, None, None,
                            ComplexityConfig(a, b, c), metadata
                        )
                        append_to_file(FAILED_FILE, failed_task)
                        failed.add(expr_str)
                        continue

                    # 6. Success!
                    config = ComplexityConfig(a, b, c)
                    task = TaskExporter.create_task(expr, x_vals, y_vals, config, metadata)
                    task["source"] = "auto_v2"
                    append_to_file(OUTPUT_FILE, task)
                    seen.add(expr_str)
                    generated += 1
                    total_generated += 1

                    logger.info(f"    [{generated}/{needed}] {expr_str[:60]}...")

                if generated < needed:
                    logger.warning(
                        f"  ⚠ Клас ({a},{b},{c}): згенеровано {generated}/{needed} "
                        f"після {attempts} спроб"
                    )

    logger.info(f"\n=== Результат ===")
    logger.info(f"Потрібно було: {total_needed}")
    logger.info(f"Згенеровано: {total_generated}")

    # Final stats
    final_counts = count_by_class(OUTPUT_FILE)
    total_tasks = sum(final_counts.values())
    min_count = min(final_counts.values()) if final_counts else 0
    max_count = max(final_counts.values()) if final_counts else 0
    logger.info(f"Загалом задач: {total_tasks}")
    logger.info(f"Мін/Макс на клас: {min_count}/{max_count}")

    under_target = sum(1 for v in final_counts.values() if v < target_per_class)
    logger.info(f"Класів < {target_per_class}: {under_target}/64")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale MCM-Gen dataset")
    parser.add_argument("--target", type=int, default=15,
                        help="Target tasks per class (default: 15)")
    parser.add_argument("--timeout", type=int, default=20,
                        help="Generation timeout in seconds (default: 20)")
    parser.add_argument("--max-attempts", type=int, default=30,
                        help="Max attempts per needed task (default: 30)")
    args = parser.parse_args()

    multiprocessing.freeze_support()
    scale_dataset(
        target_per_class=args.target,
        gen_timeout=args.timeout,
        max_attempts_per_task=args.max_attempts
    )
