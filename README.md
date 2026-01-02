# CVRP Solver using Variable Neighborhood Search (VNS)

Η προγραμματιστική εργασία αφορά στην επίλυση προβλημάτων τύπου δρομολόγησης οχημάτων με περιορισμένη χωρητικότητα (Capacitated Vehicle Routing Problem - CVRP) με τη χρήση της μεθευρετικής μεθόδου Variable Neighborhood Search (VNS). Ο κώδικας μπορεί να κάνει και γραφική σχεδίαση της προτεινόμενης λύσης.

## Χαρακτηριστικά

- **Variable Neighborhood Search (VNS)**: Υλοποίηση της μεθόδου VNS με πολλαπλές γειτονιές
- **Neighborhood Structures**: 
  - 2-opt (reversal of route segments)
  - Relocate (moving customers between routes)
  - Swap (swapping customers between routes)
  - Or-opt (relocating chains of customers)
- **VRPLIB Support**: Φόρτωση instances από VRPLIB format
- **Visualization**: Γραφική απεικόνιση των διαδρομών με matplotlib
- **Benchmarking**: Σύγκριση με best-known solutions

## Quick Start

### Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd project

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Benchmarks

```bash
# Run benchmark on all CVRP instances (100 instances)
python benchmark.py

# Run benchmark on small test set (3 instances)
python benchmark.py --instances instances/cvrp_benchmark

# Custom parameters
python benchmark.py --instances instances/cvrp --iterations 2000 --seed 42
```

Results will be saved in `visualizations/` folder with:
- Comparison plots for each instance
- Summary report (`benchmark_summary.txt`)

### Solve Single Instance

```bash
# Solve with example instance
python main.py --example

# Solve with specific instance
python main.py --instance instances/cvrp/X-n101-k25.vrp
```

## Εγκατάσταση

```bash
pip install -r requirements.txt
```

## Χρήση

### Βασική επίλυση CVRP με VNS

```bash
# Με example instance
python main.py --example

# Με VRPLIB instance
python main.py --instance path/to/instance.vrp

# Με custom parameters
python main.py --instance instance.vrp --iterations 2000 --seed 42
```

### Benchmarking - Σύγκριση με Best-known Solutions

Το benchmark script εκτελεί VNS σε όλα τα instances και συγκρίνει με best-known solutions:

```bash
# Default: Run on all 100 CVRP instances
python benchmark.py

# Quick test on 3 instances
python benchmark.py --instances instances/cvrp_benchmark

# Custom output directory
python benchmark.py --output my_results
```

**Output:**
- `visualizations/` folder: Comparison plots για κάθε instance
- `visualizations/benchmark_summary.txt`: Summary report με gaps από best-known solutions

Το benchmark script:
- Φορτώνει όλα τα instances από τον καθορισμένο φάκελο
- Εκτελεί VNS solver για κάθε instance
- Συγκρίνει με best-known solutions (αν διαθέσιμα)
- Υπολογίζει gap percentage από optimal
- Αποθηκεύει γραφήματα και summary report

### Παράμετροι main.py

- `--instance, -i`: Path προς VRPLIB instance file
- `--example, -e`: Χρήση built-in example instance
- `--iterations, -it`: Μέγιστος αριθμός iterations (default: 1000)
- `--no-improvement, -ni`: Μέγιστος αριθμός iterations χωρίς βελτίωση (default: 100)
- `--seed, -s`: Random seed για reproducibility
- `--output, -o`: Output directory για αποθήκευση plots (default: visualizations/)
- `--no-plot, -np`: Να μην αποθηκευτεί το plot

### Παράμετροι benchmark.py

- `--instances, -i`: Directory με CVRP instances (default: instances/cvrp)
- `--output, -o`: Output directory για visualizations (default: visualizations/)
- `--iterations, -it`: Max iterations για VNS (default: 2000)
- `--no-improvement, -ni`: Max iterations without improvement (default: 200)
- `--seed, -s`: Random seed (default: 42)

## Δομή Project

```
project/
├── main.py              # VNS solver για single instances
├── benchmark.py         # Benchmark script για σύγκριση
├── cvrp.py             # VNS implementation
├── vrplib_loader.py    # Instance loader
├── visualization.py     # Plotting functions
├── requirements.txt     # Dependencies
├── instances/
│   ├── cvrp/           # 100 CVRP instances + solutions
│   └── cvrp_benchmark/ # 3 test instances
└── visualizations/     # Output folder για plots
```

### Αρχεία

- `cvrp.py`: Κύρια υλοποίηση του VNS solver και CVRP data structures
- `vrplib_loader.py`: Loader για VRPLIB format instances (με υποστήριξη vrplib package)
- `visualization.py`: Μονάδα γραφικής απεικόνισης με comparison support
- `main.py`: Κύριο script για επίλυση CVRP με VNS
- `benchmark.py`: Script για benchmarking και σύγκριση με best-known solutions
- `visualizations/`: Φάκελος με όλα τα γραφήματα

## Αλγόριθμος VNS

Η υλοποίηση ακολουθεί το βασικό σχήμα του VNS:

1. **Initial Solution**: Δημιουργία αρχικής λύσης με nearest neighbor heuristic
2. **Local Search**: Εφαρμογή local search με multiple neighborhood structures
3. **Shaking**: Τυχαίες μετακινήσεις για διαφυγή από local optima
4. **Neighborhood Change**: Μεταβολή μεταξύ διαφορετικών γειτονιών

## Παράδειγμα Κώδικα

```python
from cvrp import VNS
from vrplib_loader import load_vrplib_instance
from visualization import plot_solution, print_solution_info

# Load instance
instance = load_vrplib_instance("instance.vrp")

# Create solver
solver = VNS(instance, max_iterations=1000, random_seed=42)

# Solve
solution = solver.solve()

# Print info and visualize
print_solution_info(solution)
plot_solution(solution)
```

## Απαιτήσεις

- Python 3.7+
- matplotlib
- numpy
- vrplib

## Πηγές

- [PyVRP Project](https://github.com/PyVRP)
- [VRPLIB Python Package](https://github.com/PyVRP/VRPLIB)
- [PyVRP Instances](https://github.com/PyVRP/Instances)
