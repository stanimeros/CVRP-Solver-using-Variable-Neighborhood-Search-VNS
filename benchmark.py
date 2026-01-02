"""
Benchmark script για σύγκριση VNS αποτελεσμάτων με best-known solutions
Συγκρίνει τη μέθοδο VNS με τις best-known solutions για τα ίδια προβλήματα CVRP
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from cvrp import VNS, Solution, CVRPInstance
from vrplib_loader import load_vrplib_instance, load_best_known_solution
from visualization import plot_solution, print_solution_info


def calculate_gap(vns_distance: float, best_distance: float) -> float:
    """Υπολογισμός gap percentage από best-known solution"""
    if best_distance == 0:
        return 0.0
    return ((vns_distance - best_distance) / best_distance) * 100.0


def run_benchmark(instances_dir: str = "instances/cvrp", 
                  output_dir: str = "visualizations",
                  max_iterations: int = 2000,
                  max_no_improvement: int = 200,
                  random_seed: int = 42):
    """
    Εκτέλεση benchmark σε όλα τα instances
    
    Args:
        instances_dir: Φάκελος με instances
        output_dir: Φάκελος για αποθήκευση αποτελεσμάτων
        max_iterations: Μέγιστος αριθμός iterations για VNS
        max_no_improvement: Μέγιστος αριθμός iterations χωρίς βελτίωση
        random_seed: Random seed για reproducibility
    """
    instances_path = Path(instances_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Εύρεση όλων των .vrp files
    vrp_files = list(instances_path.glob("*.vrp"))
    
    if not vrp_files:
        print(f"Δεν βρέθηκαν .vrp files στο {instances_dir}")
        print("Βεβαιωθείτε ότι έχετε instances στον φάκελο instances/")
        return
    
    print("="*70)
    print("BENCHMARK: VNS vs BEST-KNOWN SOLUTIONS")
    print("="*70)
    print(f"Instances directory: {instances_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(vrp_files)} instances")
    print("="*70)
    
    results = []
    
    for idx, vrp_file in enumerate(vrp_files, 1):
        print(f"\n[{idx}/{len(vrp_files)}] Processing: {vrp_file.name}")
        print("-" * 70)
        
        try:
            # Φόρτωση instance
            instance = load_vrplib_instance(str(vrp_file))
            print(f"Instance: {instance.name}")
            print(f"Customers: {len(instance.customers)}")
            print(f"Capacity: {instance.capacity}")
            
            # Φόρτωση best-known solution (αν υπάρχει)
            sol_file = instances_path / f"{vrp_file.stem}.sol"
            best_solution_data = None
            best_distance = None
            
            if sol_file.exists():
                best_solution_data = load_best_known_solution(str(sol_file))
                if best_solution_data:
                    # Extract distance/cost
                    if 'cost' in best_solution_data:
                        best_distance = float(best_solution_data['cost'])
                    elif 'distance' in best_solution_data:
                        best_distance = float(best_solution_data['distance'])
                    print(f"Best-known distance: {best_distance:.2f}")
            
            # Επίλυση με VNS
            print("\nΕπίλυση με VNS...")
            solver = VNS(
                instance=instance,
                max_iterations=max_iterations,
                max_no_improvement=max_no_improvement,
                random_seed=random_seed
            )
            
            vns_solution = solver.solve()
            
            # Υπολογισμός gap
            gap = None
            if best_distance is not None:
                gap = calculate_gap(vns_solution.total_distance, best_distance)
            
            # Αποθήκευση αποτελεσμάτων
            result = {
                'instance_name': instance.name,
                'instance_file': vrp_file.name,
                'customers': len(instance.customers),
                'vns_distance': vns_solution.total_distance,
                'vns_routes': len(vns_solution.routes),
                'best_distance': best_distance,
                'gap_percent': gap,
                'feasible': vns_solution.is_feasible(),
                'instance': instance,
                'solution': vns_solution
            }
            results.append(result)
            
            # Εκτύπωση αποτελεσμάτων
            print(f"\nVNS Results:")
            print(f"  Distance: {vns_solution.total_distance:.2f}")
            print(f"  Routes: {len(vns_solution.routes)}")
            print(f"  Feasible: {vns_solution.is_feasible()}")
            
            if gap is not None:
                print(f"  Gap from best-known: {gap:.2f}%")
                if gap < 5:
                    print("  ✓ Excellent!")
                elif gap < 10:
                    print("  ✓ Good!")
                elif gap < 20:
                    print("  ~ Acceptable")
                else:
                    print("  ⚠ Needs improvement")
            
            # Αποθήκευση γραφήματος
            title = f"{instance.name} - VNS Solution"
            if best_distance is not None:
                title += f"\nVNS Distance: {vns_solution.total_distance:.2f} | Best-known: {best_distance:.2f} | Gap: {gap:.2f}%"
            
            fig, ax = plot_solution(
                vns_solution,
                title=title,
                show_labels=len(instance.customers) < 50,  # Labels only for smaller instances
                figsize=(14, 10)
            )
            
            plot_filename = f"{instance.name}_comparison.png"
            plot_path = output_path / plot_filename
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  ✓ Plot saved: {plot_path}")
            
        except Exception as e:
            print(f"  ✗ Error processing {vrp_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Δημιουργία summary report
    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)
    
    summary_path = output_path / "benchmark_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("CVRP BENCHMARK RESULTS - VNS vs BEST-KNOWN SOLUTIONS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total instances: {len(results)}\n")
        f.write(f"VNS Parameters:\n")
        f.write(f"  Max iterations: {max_iterations}\n")
        f.write(f"  Max no improvement: {max_no_improvement}\n")
        f.write(f"  Random seed: {random_seed}\n\n")
        
        f.write("-"*70 + "\n")
        f.write(f"{'Instance':<30} {'Customers':<12} {'VNS Dist':<12} {'Best Dist':<12} {'Gap %':<10} {'Routes':<8}\n")
        f.write("-"*70 + "\n")
        
        total_gap = 0
        instances_with_best = 0
        
        for result in results:
            gap_str = f"{result['gap_percent']:.2f}%" if result['gap_percent'] is not None else "N/A"
            best_str = f"{result['best_distance']:.2f}" if result['best_distance'] else "N/A"
            
            f.write(f"{result['instance_name']:<30} "
                   f"{result['customers']:<12} "
                   f"{result['vns_distance']:<12.2f} "
                   f"{best_str:<12} "
                   f"{gap_str:<10} "
                   f"{result['vns_routes']:<8}\n")
            
            if result['gap_percent'] is not None:
                total_gap += result['gap_percent']
                instances_with_best += 1
        
        f.write("-"*70 + "\n")
        
        if instances_with_best > 0:
            avg_gap = total_gap / instances_with_best
            f.write(f"\nAverage gap from best-known: {avg_gap:.2f}%\n")
            f.write(f"Instances with best-known solutions: {instances_with_best}/{len(results)}\n")
        
        f.write(f"\nAll solutions feasible: {all(r['feasible'] for r in results)}\n")
    
    # Εκτύπωση summary
    print(f"\n{'Instance':<30} {'Customers':<12} {'VNS Dist':<12} {'Best Dist':<12} {'Gap %':<10}")
    print("-"*70)
    
    total_gap = 0
    instances_with_best = 0
    
    for result in results:
        gap_str = f"{result['gap_percent']:.2f}%" if result['gap_percent'] is not None else "N/A"
        best_str = f"{result['best_distance']:.2f}" if result['best_distance'] else "N/A"
        
        print(f"{result['instance_name']:<30} "
              f"{result['customers']:<12} "
              f"{result['vns_distance']:<12.2f} "
              f"{best_str:<12} "
              f"{gap_str:<10}")
        
        if result['gap_percent'] is not None:
            total_gap += result['gap_percent']
            instances_with_best += 1
    
    print("-"*70)
    
    if instances_with_best > 0:
        avg_gap = total_gap / instances_with_best
        print(f"\nAverage gap from best-known: {avg_gap:.2f}%")
    
    print(f"\n✓ Summary report saved: {summary_path}")
    print(f"✓ All plots saved in: {output_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run CVRP benchmark with VNS')
    parser.add_argument('--instances', '-i', type=str, default='instances/cvrp',
                       help='Directory with CVRP instances')
    parser.add_argument('--output', '-o', type=str, default='visualizations',
                       help='Output directory for visualizations (default: visualizations/)')
    parser.add_argument('--iterations', '-it', type=int, default=2000,
                       help='Max iterations for VNS')
    parser.add_argument('--no-improvement', '-ni', type=int, default=200,
                       help='Max iterations without improvement')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    run_benchmark(
        instances_dir=args.instances,
        output_dir=args.output,
        max_iterations=args.iterations,
        max_no_improvement=args.no_improvement,
        random_seed=args.seed
    )

