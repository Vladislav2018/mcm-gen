import json
import pandas as pd
import sympy as sp
import collections
from typing import List, Dict, Set, Tuple
from .config import OP_SETS

class DatasetAnalyzer:
    def __init__(self, benchmark_file: str, hanging_file: str):
        self.benchmark_file = benchmark_file
        self.hanging_file = hanging_file
        self.df = pd.DataFrame()
        self.stats = {}

    def load_data(self):
        """Завантажує дані з обох файлів у єдиний DataFrame."""
        data = []
        
        # Завантаження успішних
        try:
            with open(self.benchmark_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    item['status'] = 'success'
                    data.append(item)
        except FileNotFoundError:
            print(f"File {self.benchmark_file} not found.")

        # Завантаження невдалих
        try:
            with open(self.hanging_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    item['status'] = 'failed'
                    # У hanging файлах може не бути prompt_data, додамо заглушку
                    if 'prompt_data' not in item: item['prompt_data'] = {}
                    data.append(item)
        except FileNotFoundError:
            print(f"File {self.hanging_file} not found.")

        self.df = pd.DataFrame(data)
        print(f"Завантажено: {len(self.df)} записів.")
        
        # Розпаковка вектора складності
        if not self.df.empty:
            self.df['a'] = self.df['complexity_vector'].apply(lambda x: x['a'])
            self.df['b'] = self.df['complexity_vector'].apply(lambda x: x['b'])
            self.df['c'] = self.df['complexity_vector'].apply(lambda x: x['c'])

    def analyze_compliance(self):
        """
        Перевіряє, чи відповідає формула заявленому вектору складності.
        Повертає DataFrame з результатами аналізу.
        """
        results = []
        
        for idx, row in self.df.iterrows():
            formula_str = row['ground_truth']['formula']
            target_b = row['b']
            target_c = row['c']
            
            # 1. Парсинг формули
            try:
                # real=True важливо для коректного аналізу
                x = sp.Symbol('x', real=True)
                expr = sp.parse_expr(formula_str, local_dict={'x': x})
            except Exception as e:
                results.append({'parse_error': True})
                continue

            # 2. Аналіз операторів (Axis B)
            used_atoms = expr.atoms(sp.Function, sp.Pow, sp.Add, sp.Mul)
            max_op_level = 0
            illegal_ops = []
            
            # Визначаємо реальний рівень кожного використаного оператора
            for atom in used_atoms:
                # atom.func повертає клас функції (наприклад, sp.sin)
                op_type = atom.func
                
                found_level = -1
                for lvl, ops in OP_SETS.items():
                    if op_type in ops:
                        found_level = lvl
                        break
                
                if found_level > -1:
                    max_op_level = max(max_op_level, found_level)
                    if found_level > target_b:
                        illegal_ops.append(str(op_type))
            
            # Перевірка на недостатню складність (Under-complexity)
            # Якщо B=3, але ми використали тільки +, -, *, це B=0
            under_complex = (max_op_level < target_b) and (target_b > 0)

            # 3. Аналіз топології (Axis C) - Евристика
            # C=0 -> має бути EmptySet сингулярностей
            # C=2 -> НЕ має бути EmptySet
            meta_sings = row['ground_truth'].get('properties', {}).get('singularities', [])
            topo_mismatch = False
            
            if target_c == 0:
                # Очікуємо відсутність сингулярностей
                if meta_sings and meta_sings != ["analysis_timeout"]:
                    topo_mismatch = True
            elif target_c == 2:
                # Очікуємо наявність сингулярностей
                if not meta_sings: 
                    topo_mismatch = True

            results.append({
                'parse_error': False,
                'max_op_level': max_op_level,
                'illegal_ops': illegal_ops,
                'has_illegal_ops': len(illegal_ops) > 0,
                'under_complex': under_complex,
                'topo_mismatch': topo_mismatch,
                'operator_count': len(used_atoms) # Проксі для Axis A (структурна складність)
            })
            
        # Об'єднуємо результати з основним DataFrame
        analysis_df = pd.DataFrame(results)
        self.df = pd.concat([self.df.reset_index(drop=True), analysis_df], axis=1)

    def get_statistics(self):
        """Генерує текстовий звіт."""
        if self.df.empty:
            return "Немає даних для аналізу."

        report = []
        report.append("=== ЗВІТ АНАЛІЗАТОРА MCM-GEN ===\n")
        
        # 1. Загальна статистика
        success_count = len(self.df[self.df['status'] == 'success'])
        fail_count = len(self.df[self.df['status'] == 'failed'])
        report.append(f"Всього завдань: {len(self.df)}")
        report.append(f"Успішні: {success_count}")
        report.append(f"Завислі/Помилкові: {fail_count}")
        
        if success_count == 0:
            return "\n".join(report)

        # Фільтруємо тільки успішні для глибокого аналізу
        sdf = self.df[self.df['status'] == 'success']

        # 2. Матриця розподілу (Heatmap у тексті)
        report.append("\n--- Розподіл завдань по матриці (A, B, C) ---")
        matrix_counts = sdf.groupby(['a', 'b', 'c']).size().reset_index(name='count')
        # Виводимо топ-10 найпопулярніших класів
        top_classes = matrix_counts.sort_values('count', ascending=False).head(10)
        report.append(top_classes.to_string(index=False))

        # 3. Аналіз валідності (Compliance)
        report.append("\n--- Аналіз відповідності вектору складності ---")
        
        # Under-complexity
        under_complex_count = sdf['under_complex'].sum()
        report.append(f"Недостатня складність (Max Op < Target B): {under_complex_count} ({under_complex_count/len(sdf)*100:.1f}%)")
        
        # Illegal Ops
        illegal_ops_count = sdf['has_illegal_ops'].sum()
        report.append(f"Використання заборонених операторів: {illegal_ops_count} ({illegal_ops_count/len(sdf)*100:.1f}%)")
        
        # Topology Mismatch
        topo_fail = sdf['topo_mismatch'].sum()
        report.append(f"Невідповідність топології (Axis C mismatch): {topo_fail} ({topo_fail/len(sdf)*100:.1f}%)")

        # 4. Статистика операторів
        report.append("\n--- Статистика операторів ---")
        all_formulas = " ".join(sdf['ground_truth'].apply(lambda x: x['formula']).tolist())
        
        # Простий підрахунок входжень назв функцій
        ops_to_track = ['sin', 'cos', 'tan', 'exp', 'log', 'Abs', 'floor', 'Piecewise', 'factorial', 'besselj', 'gamma', 'erf', 'zeta']
        op_stats = {op: all_formulas.count(op) for op in ops_to_track}
        sorted_ops = sorted(op_stats.items(), key=lambda x: x[1], reverse=True)
        
        for op, count in sorted_ops:
            report.append(f"{op}: {count}")

        return "\n".join(report)

    def export_csv(self, filename="analysis_report.csv"):
        self.df.to_csv(filename, index=False)