import json
import os
import numpy as np
from src.config import ComplexityConfig
from src.generator import ExpressionGenerator
from src.validator import TopologyFilter
from src.sampler import DatasetSampler, TaskExporter

def generate_benchmark_suite(filename="benchmark_tasks.jsonl", tasks_per_class=2):
    if os.path.exists(filename): os.remove(filename)

    with open(filename, "a", encoding="utf-8") as f:
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    print(f"Генеруємо <{a},{b},{c}>...", end=" ", flush=True)
                    config = ComplexityConfig(a, b, c)
                    gen = ExpressionGenerator(config)
                    validator = TopologyFilter(config)
                    
                    count = 0
                    attempts = 0
                    while count < tasks_per_class and attempts < 30: # Обмеження спроб
                        attempts += 1
                        expr = gen.generate()
                        if validator.check(expr):
                            x, y, meta = DatasetSampler.get_data(expr, n_points=25)
                            if x is not None:
                                task = TaskExporter.create_task(expr, x, y, config, meta)
                                f.write(json.dumps(task, ensure_ascii=False) + "\n")
                                count += 1
                    print(f"Готово ({count} шт)")

if __name__ == "__main__":
    generate_benchmark_suite()