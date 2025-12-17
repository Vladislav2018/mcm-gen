import json
import os
from src.config import ComplexityConfig
from src.generator import ExpressionGenerator
from src.validator import TopologyFilter
from src.sampler import DatasetSampler, TaskExporter

def generate_benchmark_suite(filename="benchmark_tasks.jsonl", tasks_per_class=2):
    """Генерує набір завдань для всіх 64 класів MCM."""
    if os.path.exists(filename):
        os.remove(filename)

    with open(filename, "a", encoding="utf-8") as f:
        # Проходимо по всіх 64 комбінаціях таксономії
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    print(f"Генерація завдань для класу <{a},{b},{c}>...")
                    config = ComplexityConfig(a, b, c)
                    gen = ExpressionGenerator(config)
                    validator = TopologyFilter(config)
                    
                    count = 0
                    while count < tasks_per_class:
                        expr = gen.generate()
                        if validator.check(expr):
                            x, y = DatasetSampler.get_data(expr)
                            if x is not None and np.all(np.isfinite(y)):
                                task = TaskExporter.create_task(expr, x, y, config)
                                f.write(json.dumps(task, ensure_ascii=False) + "\n")
                                count += 1

if __name__ == "__main__":
    generate_benchmark_suite()
    print("\nБенчмарк успішно згенеровано у файл benchmark_tasks.jsonl")