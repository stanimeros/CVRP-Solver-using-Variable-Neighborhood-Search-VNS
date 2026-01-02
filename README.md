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

```bash
python benchmark.py --instances instances/cvrp --output visualizations
```

Το benchmark script:
- Φορτώνει όλα τα instances από τον καθορισμένο φάκελο
- Εκτελεί VNS solver για κάθε instance
- Συγκρίνει με best-known solutions (αν διαθέσιμα)
- Αποθηκεύει γραφήματα στο `visualizations/` folder
- Δημιουργεί summary report με gaps

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
