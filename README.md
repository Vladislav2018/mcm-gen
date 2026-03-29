# MCM-Gen: Multidimensional Complexity Matrix Benchmark Generator

MCM-Gen is a robust, procedural generation framework designed to create high-quality datasets for evaluating the mathematical reasoning capabilities of Large Language Models (LLMs) in Symbolic Regression tasks.

MCM-Gen combats the growing problem of "data contamination" in static benchmarks (like Feynman or SRBench) by generating a geometrically endless stream of novel mathematical expressions governed by a strict theoretical taxonomy.

## The MCM Taxonomy (A, B, C)

Every benchmark task is categorized across three orthogonal axes, forming a $4 \times 4 \times 4 = 64$ class matrix:

1. **Structural Complexity (Axis A)**: Focuses on the depth of the computational graph.
   - **A0**: Depth 1 (Linear/Simple expressions)
   - **A1**: Depth 2 (Quadratic/Moderate nesting)
   - **A2**: Depth 3 (Complex nesting)
   - **A3**: Depth 4+ (Deep compositional graphs)

2. **Semantic Complexity (Axis B)**: Defines the allowed mathematical operator set.
   - **B0**: Arithmetics & Polynomials (`+`, `-`, `*`, `^`)
   - **B1**: Transcendental (`sin`, `cos`, `tan`, `exp`, `log`)
   - **B2**: Non-smooth (`Abs`, `floor`, `Piecewise`, `sign`)
   - **B3**: Special Functions (`besselj`, `gamma`, `erf`, `zeta`)

3. **Topological Feature Salience (Axis C)**: Evaluates the function's qualitative behavior.
   - **C0**: Regular (Smooth, finite, non-periodic)
   - **C1**: Periodic (Cyclic behavior)
   - **C2**: Isolated Singularities (Vertical asymptotes, poles)
   - **C3**: Complex/Chaotic (Oscillating singularities, multiple poles, rapid changes)

## Components

The framework uses a Reverse-Flow Generation approach (Formula $\rightarrow$ Data) to guarantee ground-truth accuracy.

* `generator.py`: A template-based expression generator. It uses parameterized mathematical templates specific to each (B,C) combination and randomizes coefficients, avoiding trivial duplicates.
* `validator.py`: A rigorous quality filter. It uses a combination of symbolic (SymPy) and numerical criteria to ensure the generated function truly matches its requested Topology (Axis C).
* `sampler.py`: Computes point pairs $(x, y)$. Features adaptive domain selection to naturally capture the essential behavior (e.g., surrounding a singularity, or capturing 2-3 full periods). Employs safe multiprocessing with timeouts to prevent SymPy solver hangs.
* `quality_audit.py`: Validates the full dataset for B-compliance, C-compliance, operator diversity, and matrix distribution coverage.

## MCM-Bench v1.0 Dataset

The current generated dataset (`data/benchmark_tasks.jsonl`) contains **818 unique tasks** covering all 64 complexity classes, distributed as:
- **Regular (C0)**: 199 tasks
- **Periodic (C1)**: 210 tasks
- **Singularities (C2)**: 201 tasks
- **Complex (C3)**: 208 tasks

*Note: Datasets for derivative and integral functions are provided in `data/benchmark_derivatives.jsonl` and `data/benchmark_integrals.jsonl`.*

## Installation & Usage

### 1. Requirements
- Python 3.9+
- `sympy`, `numpy`, `pandas`

### 2. Generating the Dataset
To scale or regenerate the dataset, run the scaling script:
```bash
python scale_dataset.py --target 15 --timeout 20
```
*Note: Due to the complexity of exact symbolic math computations, generation is CPU-intensive. Generating 15 tasks per class takes approx 1-2 hours.*

### 3. Quality Audit
To verify the compliance and diversity metrics of the generated benchmark:
```bash
python quality_audit.py > final_audit_report.txt
```

## Citation
If you use MCM-Gen or the associated benchmark, please cite Paper 1 (Theoretical Framework) and Paper 2 (Dataset and Generation Validation).