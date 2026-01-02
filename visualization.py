"""
Visualization module for CVRP solutions
"""

import matplotlib.pyplot as plt
from typing import List, Optional
from cvrp import Solution
import numpy as np


def plot_solution(solution: Solution, title: str = "CVRP Solution", 
                  show_labels: bool = True, figsize: tuple = (12, 8)):
    """
    Plot a CVRP solution showing routes and customers
    
    Args:
        solution: The solution to visualize
        title: Plot title
        show_labels: Whether to show customer IDs
        figsize: Figure size
    """
    instance = solution.instance
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate colors for routes
    colors = plt.cm.tab20(np.linspace(0, 1, len(solution.routes)))
    
    # Plot depot
    ax.scatter(instance.depot.x, instance.depot.y, 
              c='red', s=200, marker='s', edgecolors='black', linewidths=2,
              label='Depot', zorder=5)
    
    if show_labels:
        ax.annotate('Depot', (instance.depot.x, instance.depot.y),
                   xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    # Plot routes
    for route_idx, route in enumerate(solution.routes):
        if not route:
            continue
        
        color = colors[route_idx]
        
        # Get coordinates for route
        route_x = [instance.depot.x]
        route_y = [instance.depot.y]
        
        for customer_id in route:
            customer = instance.customers[customer_id]
            route_x.append(customer.x)
            route_y.append(customer.y)
        
        route_x.append(instance.depot.x)
        route_y.append(instance.depot.y)
        
        # Plot route line
        ax.plot(route_x, route_y, color=color, linewidth=2, alpha=0.6, 
               label=f'Route {route_idx + 1} ({len(route)} customers)')
        
        # Plot customers
        for customer_id in route:
            customer = instance.customers[customer_id]
            ax.scatter(customer.x, customer.y, c=[color], s=100, 
                      edgecolors='black', linewidths=1, zorder=4)
            
            if show_labels:
                ax.annotate(f'{customer_id}', (customer.x, customer.y),
                           xytext=(3, 3), textcoords='offset points', fontsize=8)
    
    # Plot unvisited customers (if any)
    visited = set()
    for route in solution.routes:
        visited.update(route)
    
    unvisited = [i for i in range(len(instance.customers)) if i not in visited]
    if unvisited:
        for customer_id in unvisited:
            customer = instance.customers[customer_id]
            ax.scatter(customer.x, customer.y, c='gray', s=100, 
                      marker='x', linewidths=2, zorder=3)
            if show_labels:
                ax.annotate(f'{customer_id}', (customer.x, customer.y),
                           xytext=(3, 3), textcoords='offset points', fontsize=8, 
                           color='red', fontweight='bold')
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(f'{title}\nTotal Distance: {solution.total_distance:.2f} | Routes: {len(solution.routes)}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig, ax


def print_solution_info(solution: Solution, best_distance: Optional[float] = None):
    """Print detailed information about the solution"""
    print("\n" + "="*60)
    print(f"Solution Information")
    print("="*60)
    print(f"Total Distance: {solution.total_distance:.2f}")
    
    if best_distance is not None:
        gap = ((solution.total_distance - best_distance) / best_distance) * 100.0
        print(f"Best-known Distance: {best_distance:.2f}")
        print(f"Gap from best-known: {gap:.2f}%")
    
    print(f"Number of Routes: {len(solution.routes)}")
    print(f"Feasible: {solution.is_feasible()}")
    print("\nRoute Details:")
    
    for route_idx, route in enumerate(solution.routes):
        route_demand = solution.get_route_demand(route)
        route_distance = 0.0
        
        if route:
            # Distance from depot to first customer
            first_customer = solution.instance.customers[route[0]]
            route_distance += solution.instance.distance(
                solution.instance.depot, first_customer)
            
            # Distance between customers
            for i in range(len(route) - 1):
                c1 = solution.instance.customers[route[i]]
                c2 = solution.instance.customers[route[i + 1]]
                route_distance += solution.instance.distance(c1, c2)
            
            # Distance from last customer to depot
            last_customer = solution.instance.customers[route[-1]]
            route_distance += solution.instance.distance(
                last_customer, solution.instance.depot)
        
        print(f"  Route {route_idx + 1}: {len(route)} customers, "
              f"demand: {route_demand:.1f}/{solution.instance.capacity:.1f}, "
              f"distance: {route_distance:.2f}")
        if route:
            print(f"    Customers: {route}")
    
    print("="*60 + "\n")

