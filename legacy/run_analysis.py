from src.analyzer import DatasetAnalyzer
import os

def main():
    print("Запуск аналізу даних...")
    
    analyzer = DatasetAnalyzer(
        benchmark_file="benchmark_tasks.jsonl",
        hanging_file="hanging_functions.jsonl"
    )
    
    analyzer.load_data()
    
    if analyzer.df.empty:
        print("Файли порожні або відсутні.")
        return

    print("Виконання інтелектуальної перевірки...")
    analyzer.analyze_compliance()
    
    print("Генерація звіту...")
    report = analyzer.get_statistics()
    print(report)
    
    # Збереження детальної таблиці (можна відкрити в Excel)
    analyzer.export_csv("analysis_full.csv")
    print("\nДетальний CSV звіт збережено як 'analysis_full.csv'")

if __name__ == "__main__":
    main()