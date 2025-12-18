import cProfile
import pstats
from main import generate_benchmark_suite

def profile_generator():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Запускаємо невеликий тест (наприклад, по 1 завданню на клас)
    generate_benchmark_suite(tasks_per_class=1)
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20) # Виведемо топ-20 найповільніших функцій

if __name__ == "__main__":
    profile_generator()