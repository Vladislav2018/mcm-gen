"""MCM-Gen LaTeX Table Generator

Creates LaTeX formatted tables for the paper based on the current benchmark data.
"""

import json
from collections import Counter, defaultdict

OUTPUT_FILE = "data/benchmark_tasks.jsonl"
TEX_FILE = "paper/paper_tables.tex"

def load_data():
    tasks = []
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks

def generate_coverage_table(tasks):
    class_counts = Counter()
    for t in tasks:
        cv = t['complexity_vector']
        class_counts[(cv['a'], cv['b'], cv['c'])] += 1

    tex = "% Table 1: Distribution of tasks across A and B axes (aggregated over C)\n"
    tex += "\\begin{table}[h]\n\\centering\n"
    tex += "\\begin{tabular}{l|cccc|c}\n"
    tex += "\\toprule\n"
    tex += "\\textbf{Structural (A)} & \\multicolumn{4}{c|}{\\textbf{Semantic (B)}} & \\textbf{Total} \\\\\n"
    tex += " & \\textbf{B=0} & \\textbf{B=1} & \\textbf{B=2} & \\textbf{B=3} & \\\\\n"
    tex += "\\midrule\n"

    for a in range(4):
        tex += f"\\textbf{{A={a}}} "
        row_total = 0
        for b in range(4):
            # Sum over c
            count = sum(class_counts[(a,b,c)] for c in range(4))
            tex += f"& {count} "
            row_total += count
        tex += f"& \\textbf{{{row_total}}} \\\\\n"
    
    tex += "\\midrule\n"
    tex += "\\textbf{Total} "
    grand_total = 0
    for b in range(4):
        col_total = sum(class_counts[(a,b,c)] for a in range(4) for c in range(4))
        tex += f"& \\textbf{{{col_total}}} "
        grand_total += col_total
    tex += f"& \\textbf{{{grand_total}}} \\\\\n"
    
    tex += "\\bottomrule\n"
    tex += "\\end{tabular}\n"
    tex += "\\caption{Distribution of generated tasks across Structural (A) and Semantic (B) complexity axes.}\n"
    tex += "\\label{tab:coverage_ab}\n"
    tex += "\\end{table}\n\n"
    return tex

def generate_c_distribution_table(tasks):
    class_counts = Counter()
    for t in tasks:
        cv = t['complexity_vector']
        class_counts[cv['c']] += 1

    tex = "% Table 2: Distribution across Topological Axis (C)\n"
    tex += "\\begin{table}[h]\n\\centering\n"
    tex += "\\begin{tabular}{llc}\n"
    tex += "\\toprule\n"
    tex += "\\textbf{C Class} & \\textbf{Description} & \\textbf{Tasks} \\\\\n"
    tex += "\\midrule\n"
    
    labels = ["Regular", "Periodic", "Isolated Singularities", "Complex/Chaotic"]
    for c in range(4):
        tex += f"\\textbf{{C={c}}} & {labels[c]} & {class_counts[c]} \\\\\n"
    
    tex += "\\bottomrule\n"
    tex += "\\end{tabular}\n"
    tex += "\\caption{Distribution of generated tasks across the Topological (C) axis.}\n"
    tex += "\\label{tab:coverage_c}\n"
    tex += "\\end{table}\n\n"
    return tex


def generate_source_table(tasks):
    sources = Counter(t.get('source', 'unknown') for t in tasks)
    
    tex = "% Table 3: Task Origins\n"
    tex += "\\begin{table}[h]\n\\centering\n"
    tex += "\\begin{tabular}{lcc}\n"
    tex += "\\toprule\n"
    tex += "\\textbf{Source} & \\textbf{Count} & \\textbf{\\%} \\\\\n"
    tex += "\\midrule\n"
    
    total = sum(sources.values())
    for s, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total) * 100
        label = s.replace('_', '\\_')
        tex += f"{label} & {count} & {pct:.1f}\\% \\\\\n"
    
    tex += "\\bottomrule\n"
    tex += "\\end{tabular}\n"
    tex += "\\caption{Origins of the tasks in the MCM-Bench dataset.}\n"
    tex += "\\label{tab:task_sources}\n"
    tex += "\\end{table}\n\n"
    return tex


def main():
    tasks = load_data()
    
    with open(TEX_FILE, 'w', encoding='utf-8') as f:
        f.write(generate_coverage_table(tasks))
        f.write(generate_c_distribution_table(tasks))
        f.write(generate_source_table(tasks))
    print(f"Generated {TEX_FILE}")

if __name__ == "__main__":
    main()
