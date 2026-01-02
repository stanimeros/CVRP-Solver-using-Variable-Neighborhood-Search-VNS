"""
Main script for solving CVRP using Variable Neighborhood Search (VNS)
Η προγραμματιστική εργασία αφορά στην επίλυση προβλημάτων τύπου δρομολόγησης οχημάτων
με περιορισμένη χωρητικότητα Capacitated Vehicle Routing Problem (CVRP) με τη χρήση της
μεθευρετικής μεθόδου Variable Neighborhood Search (VNS).
"""

import argparse
import sys
import os
from pathlib import Path
from cvrp import VNS, CVRPInstance
from vrplib_loader import load_vrplib_instance, create_simple_instance
from visualization import plot_solution, print_solution_info
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt


def create_example_instance():
    """Create a simple example instance for testing"""
    # Example: 10 customers around a depot
    customers_data = [
        (10, 10, 5),   # Customer 0
        (20, 15, 8),   # Customer 1
        (15, 25, 6),   # Customer 2
        (30, 20, 9),   # Customer 3
        (25, 30, 7),   # Customer 4
        (40, 25, 5),   # Customer 5
        (35, 35, 8),   # Customer 6
        (50, 30, 6),   # Customer 7
        (45, 40, 7),   # Customer 8
        (60, 35, 9),   # Customer 9
    ]
    
    return create_simple_instance(
        name="Example_10",
        customers_data=customers_data,
        capacity=20.0,
        depot=(0, 0)
    )


def main():
    parser = argparse.ArgumentParser(
        description='Solve CVRP using Variable Neighborhood Search (VNS)'
    )
    parser.add_argument(
        '--instance', '-i',
        type=str,
        help='Path to VRPLIB instance file'
    )
    parser.add_argument(
        '--example', '-e',
        action='store_true',
        help='Use built-in example instance'
    )
    parser.add_argument(
        '--iterations', '-it',
        type=int,
        default=1000,
        help='Maximum number of iterations (default: 1000)'
    )
    parser.add_argument(
        '--no-improvement', '-ni',
        type=int,
        default=100,
        help='Maximum iterations without improvement (default: 100)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='visualizations',
        help='Output directory for saving the plot (default: visualizations/)'
    )
    parser.add_argument(
        '--no-plot', '-np',
        action='store_true',
        help='Do not save the plot'
    )
    
    args = parser.parse_args()
    
    # Load instance
    if args.example:
        print("Using built-in example instance...")
        instance = create_example_instance()
    elif args.instance:
        print(f"Loading instance from {args.instance}...")
        try:
            instance = load_vrplib_instance(args.instance)
        except Exception as e:
            print(f"Error loading instance: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("No instance specified. Using built-in example...")
        instance = create_example_instance()
    
    print(f"\nInstance: {instance.name}")
    print(f"Customers: {len(instance.customers)}")
    print(f"Vehicle Capacity: {instance.capacity}")
    print(f"Total Demand: {sum(c.demand for c in instance.customers)}")
    
    # Create solver
    solver = VNS(
        instance=instance,
        max_iterations=args.iterations,
        max_no_improvement=args.no_improvement,
        random_seed=args.seed
    )
    
    print(f"\nSolving with VNS...")
    print(f"Max iterations: {args.iterations}")
    print(f"Max no improvement: {args.no_improvement}")
    
    # Solve
    solution = solver.solve()
    
    # Print solution info
    print_solution_info(solution)
    
    # Visualize and save
    if not args.no_plot:
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create plot
        fig, ax = plot_solution(solution, title=f"CVRP Solution - {instance.name}")
        
        # Save plot
        filename = f"{instance.name}_solution.png"
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n✓ Plot saved to {filepath}")
    
    return solution


if __name__ == "__main__":
    main()

