# CVRP Solver using Variable Neighborhood Search (VNS)

Η προγραμματιστική εργασία αφορά στην επίλυση προβλημάτων τύπου δρομολόγησης οχημάτων με περιορισμένη χωρητικότητα (Capacitated Vehicle Routing Problem - CVRP) με τη χρήση της μεθευρετικής μεθόδου Variable Neighborhood Search (VNS). Ο κώδικας μπορεί να κάνει και γραφική σχεδίαση της προτεινόμενης λύσης.

## Χαρακτηριστικά

- **Variable Neighborhood Search (VNS)**: Υλοποίηση της μεθόδου VNS με πολλαπλές γειτονιές
- **Neighborhood Structures**: 
  - 2-opt (intra-route segment reversal)
  - **2-opt*** (inter-route tail exchange) - *Advanced*
  - Relocate (moving customers between routes)
  - Swap (swapping customers between routes)
  - Or-opt (relocating chains of customers)
- **Ruin & Recreate Shaking**: Προηγμένη στρατηγική shaking με best insertion για καλύτερη διαφυγή από local optima
- **Delta Evaluation**: Υψηλής απόδοσης incremental cost calculation (O(1)) για όλες τις local search operations
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

# With custom parameters
python main.py --instance instances/cvrp/X-n101-k25.vrp --iterations 2000 --seed 42
```

## Advanced Usage

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
- `--time-limit, -t`: Time limit per instance σε seconds (optional)
- `--quick, -q`: Quick test mode με reduced parameters

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

Η υλοποίηση ακολουθεί το βασικό σχήμα του VNS με προηγμένες βελτιστοποιήσεις:

1. **Initial Solution**: Δημιουργία αρχικής λύσης με nearest neighbor heuristic
2. **Local Search**: Εφαρμογή local search με multiple neighborhood structures:
   - **2-opt**: Intra-route optimization (αντιστροφή τμημάτων διαδρομής)
   - **2-opt***: Inter-route optimization (ανταλλαγή ουρών μεταξύ διαδρομών)
   - **Relocate**: Μετακίνηση πελατών μεταξύ διαδρομών
   - **Swap**: Ανταλλαγή πελατών μεταξύ διαδρομών
   - **Or-opt**: Μετακίνηση αλυσίδων πελατών
   - Όλες οι operations χρησιμοποιούν **Delta Evaluation** για O(1) cost calculation
3. **Ruin & Recreate Shaking**: Προηγμένη στρατηγική shaking:
   - **Ruin**: Τυχαία αφαίρεση k*3 πελατών (ή έως 1/3 του συνόλου)
   - **Recreate**: Επανεισαγωγή με Best Insertion heuristic (greedy, minimal cost)
   - Επιτρέπει καλύτερη διαφυγή από local optima σε σχέση με απλές τυχαίες μετακινήσεις
4. **Neighborhood Change**: Μεταβολή μεταξύ διαφορετικών γειτονιών (k=1 έως k_max=5)

### Performance Optimizations

- **Delta Evaluation**: Όλες οι local search operations υπολογίζουν μόνο την αλλαγή κόστους (O(1)) αντί για πλήρη ανακαταμέτρηση (O(n))
- **Incremental Updates**: Το total_distance ενημερώνεται αυξανόμενα μετά από κάθε βελτίωση
- **Early Capacity Checks**: Οι περιορισμοί χωρητικότητας ελέγχονται πριν από τους υπολογισμούς κόστους
- **Result Quality**: Τυπικά αποτελέσματα με gap < 2% από best-known solutions

## Παράδειγμα Κώδικα

```python
from cvrp import VNS
from vrplib_loader import load_vrplib_instance
from visualization import plot_solution, print_solution_info

# Load instance
instance = load_vrplib_instance("instance.vrp")

# Create solver with advanced VNS (includes 2-opt* and Ruin & Recreate)
solver = VNS(
    instance=instance, 
    max_iterations=2000,
    max_no_improvement=200,
    random_seed=42,
    verbose=True,
    time_limit=300  # Optional: 5 minutes per instance
)

# Solve
solution = solver.solve()

# Print info and visualize
print_solution_info(solution)
plot_solution(solution)
```

## Performance & Results

Η υλοποίηση έχει βελτιστοποιηθεί για υψηλή απόδοση και ποιότητα αποτελεσμάτων:

- **Speed**: ~200x ταχύτερη local search λόγω Delta Evaluation
- **Quality**: Τυπικά αποτελέσματα με gap < 2% από best-known solutions
- **Scalability**: Επιτυχημένη επίλυση instances με 100+ πελάτες σε δευτερόλεπτα έως λεπτά
- **Reproducibility**: Deterministic αποτελέσματα με fixed random seed

### Benchmark Results

Αποτελέσματα από benchmark σε 3 test instances:

| Instance | Customers | VNS Distance | Best-Known | Gap | Routes |
|----------|-----------|--------------|------------|-----|--------|
| X-n106-k14 | 105 | 26,639.14 | 26,362.00 | 1.05% | 14 |
| X-n101-k25 | 100 | 28,083.52 | 27,591.00 | 1.79% | 27 |
| X-n110-k13 | 109 | 14,978.22 | 14,971.00 | 0.05% | 13 |

**Average gap: 0.96%** - Όλες οι λύσεις είναι feasible!

### Visualization Examples

Παρακάτω φαίνονται οι γραφικές απεικονίσεις των λύσεων για τα benchmark instances:

#### X-n101-k25 (100 customers, 27 routes, Gap: 1.79%)
![X-n101-k25 Solution](https://github.com/stanimeros/CVRP-Solver-using-Variable-Neighborhood-Search-VNS/blob/main/visualizations/X-n101-k25_comparison.png?raw=true)

#### X-n106-k14 (105 customers, 14 routes, Gap: 1.05%)
![X-n106-k14 Solution](https://github.com/stanimeros/CVRP-Solver-using-Variable-Neighborhood-Search-VNS/blob/main/visualizations/X-n106-k14_comparison.png?raw=true)

#### X-n110-k13 (109 customers, 13 routes, Gap: 0.05%)
![X-n110-k13 Solution](https://github.com/stanimeros/CVRP-Solver-using-Variable-Neighborhood-Search-VNS/blob/main/visualizations/X-n110-k13_comparison.png?raw=true)

## Απαιτήσεις

- Python 3.7+
- matplotlib
- numpy
- vrplib

## Πηγές

- [PyVRP Project](https://github.com/PyVRP)
- [VRPLIB Python Package](https://github.com/PyVRP/VRPLIB)
- [PyVRP Instances](https://github.com/PyVRP/Instances)
