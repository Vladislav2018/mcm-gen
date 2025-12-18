import logging
import sys
import json
import os
import multiprocessing
import sympy as sp
from typing import Set, Dict, Any

def setup_logging():
    logger = logging.getLogger("MCM-Gen")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Очистка попередніх хендлерів, щоб уникнути дублювання
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler("mcm_gen.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def load_manual_formulas(filepath: str) -> Dict[str, list]:
    """Завантажує ручні формули з JSON файлу."""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_seen_expressions(filepath: str) -> Set[str]:
    """Зчитує вже згенеровані вирази з існуючого файлу результатів."""
    seen = set()
    if not os.path.exists(filepath):
        return seen
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Нормалізуємо формулу, щоб уникнути дублікатів через пробіли
                formula = data.get("ground_truth", {}).get("formula", "")
                if formula:
                    # Спроба спростити через SymPy для канонічного вигляду (опціонально, бо довго)
                    # Для швидкості просто беремо string, або strip()
                    seen.add(formula)
            except json.JSONDecodeError:
                continue
    return seen

def append_to_file(filepath: str, data: Dict[str, Any]):
    """Дописує один об'єкт JSONL у файл."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

# --- TIMEOUT INFRASTRUCTURE ---

def _worker(func, args, queue):
    """Допоміжний воркер для запуску функції в окремому процесі."""
    try:
        result = func(*args)
        queue.put((True, result))
    except Exception as e:
        queue.put((False, str(e)))

def run_with_timeout(func, args=(), timeout=5):
    """
    Запускає функцію в окремому процесі з обмеженням часу.
    Повертає (success, result/error_message).
    Якщо timeout - success=False, error="Timeout".
    """
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(func, args, queue))
    p.start()
    
    p.join(timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return False, "Timeout"
    
    if not queue.empty():
        return queue.get()
    
    return False, "Unknown Error (Worker died)"