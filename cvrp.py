"""
Capacitated Vehicle Routing Problem (CVRP) Solver using Variable Neighborhood Search (VNS)
"""

import math
import random
import copy
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Customer:
    """Represents a customer in the CVRP"""
    id: int
    x: float
    y: float
    demand: float


@dataclass
class CVRPInstance:
    """Represents a CVRP problem instance"""
    name: str
    dimension: int
    capacity: float
    depot: Customer
    customers: List[Customer]
    
    def distance(self, c1: Customer, c2: Customer) -> float:
        """Calculate Euclidean distance between two customers"""
        return math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)


class Solution:
    """Represents a solution to the CVRP"""
    
    def __init__(self, instance: CVRPInstance):
        self.instance = instance
        self.routes: List[List[int]] = []  # List of routes, each route is a list of customer IDs
        self.total_distance: float = 0.0
        
    def calculate_distance(self):
        """Calculate total distance of the solution"""
        self.total_distance = 0.0
        for route in self.routes:
            if not route:
                continue
            # Distance from depot to first customer
            first_customer = self.instance.customers[route[0]]
            self.total_distance += self.instance.distance(self.instance.depot, first_customer)
            
            # Distance between consecutive customers
            for i in range(len(route) - 1):
                c1 = self.instance.customers[route[i]]
                c2 = self.instance.customers[route[i + 1]]
                self.total_distance += self.instance.distance(c1, c2)
            
            # Distance from last customer to depot
            last_customer = self.instance.customers[route[-1]]
            self.total_distance += self.instance.distance(last_customer, self.instance.depot)
    
    def is_feasible(self) -> bool:
        """Check if solution is feasible (capacity constraints)"""
        for route in self.routes:
            total_demand = sum(self.instance.customers[cid].demand for cid in route)
            if total_demand > self.instance.capacity:
                return False
        return True
    
    def get_route_demand(self, route: List[int]) -> float:
        """Get total demand of a route"""
        return sum(self.instance.customers[cid].demand for cid in route)
    
    def copy(self):
        """Create a deep copy of the solution"""
        new_solution = Solution(self.instance)
        new_solution.routes = [route.copy() for route in self.routes]
        new_solution.total_distance = self.total_distance
        return new_solution


class VNS:
    """Variable Neighborhood Search solver for CVRP"""
    
    def __init__(self, instance: CVRPInstance, max_iterations: int = 1000, 
                 max_no_improvement: int = 100, random_seed: Optional[int] = None,
                 verbose: bool = True, time_limit: Optional[float] = None):
        self.instance = instance
        self.max_iterations = max_iterations
        self.max_no_improvement = max_no_improvement
        self.verbose = verbose
        self.time_limit = time_limit  # Time limit in seconds
        if random_seed is not None:
            random.seed(random_seed)
    
    def initial_solution(self) -> Solution:
        """Generate initial solution using nearest neighbor heuristic"""
        solution = Solution(self.instance)
        unvisited = list(range(len(self.instance.customers)))
        random.shuffle(unvisited)
        
        current_route = []
        current_load = 0.0
        
        while unvisited:
            if not current_route:
                # Start new route
                customer_id = unvisited.pop(0)
                current_route.append(customer_id)
                current_load = self.instance.customers[customer_id].demand
            else:
                # Find nearest unvisited customer
                last_customer = self.instance.customers[current_route[-1]]
                nearest = None
                nearest_dist = float('inf')
                
                for cid in unvisited:
                    customer = self.instance.customers[cid]
                    dist = self.instance.distance(last_customer, customer)
                    if dist < nearest_dist and current_load + customer.demand <= self.instance.capacity:
                        nearest_dist = dist
                        nearest = cid
                
                if nearest is not None:
                    current_route.append(nearest)
                    current_load += self.instance.customers[nearest].demand
                    unvisited.remove(nearest)
                else:
                    # Cannot add more customers, start new route
                    solution.routes.append(current_route)
                    current_route = []
                    current_load = 0.0
        
        if current_route:
            solution.routes.append(current_route)
        
        solution.calculate_distance()
        return solution
    
    def two_opt(self, solution: Solution, route_idx: int) -> bool:
        """2-opt neighborhood: reverse a segment of a route"""
        route = solution.routes[route_idx]
        if len(route) < 2:
            return False
        
        improved = False
        best_solution = solution.copy()
        
        for i in range(len(route)):
            for j in range(i + 2, len(route)):
                # Create new route by reversing segment [i+1:j+1]
                new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                new_solution = solution.copy()
                new_solution.routes[route_idx] = new_route
                new_solution.calculate_distance()
                
                if new_solution.total_distance < best_solution.total_distance:
                    best_solution = new_solution
                    improved = True
        
        if improved:
            solution.routes = best_solution.routes
            solution.total_distance = best_solution.total_distance
        return improved
    
    def relocate(self, solution: Solution) -> bool:
        """Relocate neighborhood: move a customer from one route to another"""
        improved = False
        best_solution = solution.copy()
        
        for r1_idx in range(len(solution.routes)):
            route1 = solution.routes[r1_idx]
            for c1_idx in range(len(route1)):
                customer_id = route1[c1_idx]
                customer = self.instance.customers[customer_id]
                
                # Try inserting in all other positions
                for r2_idx in range(len(solution.routes)):
                    if r1_idx == r2_idx:
                        continue
                    
                    route2 = solution.routes[r2_idx]
                    route2_demand = solution.get_route_demand(route2)
                    
                    if route2_demand + customer.demand > self.instance.capacity:
                        continue
                    
                    for pos in range(len(route2) + 1):
                        new_solution = solution.copy()
                        # Remove customer from route1
                        new_solution.routes[r1_idx] = route1[:c1_idx] + route1[c1_idx+1:]
                        # Insert customer in route2
                        new_solution.routes[r2_idx] = route2[:pos] + [customer_id] + route2[pos:]
                        
                        # Remove empty routes
                        new_solution.routes = [r for r in new_solution.routes if r]
                        
                        new_solution.calculate_distance()
                        
                        if new_solution.total_distance < best_solution.total_distance:
                            best_solution = new_solution
                            improved = True
        
        if improved:
            solution.routes = best_solution.routes
            solution.total_distance = best_solution.total_distance
        return improved
    
    def swap(self, solution: Solution) -> bool:
        """Swap neighborhood: swap two customers between routes"""
        improved = False
        best_solution = solution.copy()
        
        for r1_idx in range(len(solution.routes)):
            route1 = solution.routes[r1_idx]
            for c1_idx in range(len(route1)):
                customer1_id = route1[c1_idx]
                customer1 = self.instance.customers[customer1_id]
                
                for r2_idx in range(r1_idx + 1, len(solution.routes)):
                    route2 = solution.routes[r2_idx]
                    route1_demand = solution.get_route_demand(route1) - customer1.demand
                    route2_demand = solution.get_route_demand(route2)
                    
                    for c2_idx in range(len(route2)):
                        customer2_id = route2[c2_idx]
                        customer2 = self.instance.customers[customer2_id]
                        
                        # Check capacity constraints
                        if (route1_demand + customer2.demand > self.instance.capacity or
                            route2_demand - customer2.demand + customer1.demand > self.instance.capacity):
                            continue
                        
                        new_solution = solution.copy()
                        # Swap customers
                        new_solution.routes[r1_idx] = route1[:c1_idx] + [customer2_id] + route1[c1_idx+1:]
                        new_solution.routes[r2_idx] = route2[:c2_idx] + [customer1_id] + route2[c2_idx+1:]
                        
                        new_solution.calculate_distance()
                        
                        if new_solution.total_distance < best_solution.total_distance:
                            best_solution = new_solution
                            improved = True
        
        if improved:
            solution.routes = best_solution.routes
            solution.total_distance = best_solution.total_distance
        return improved
    
    def or_opt(self, solution: Solution) -> bool:
        """Or-opt neighborhood: relocate a chain of 2-3 consecutive customers"""
        improved = False
        best_solution = solution.copy()
        
        for r1_idx in range(len(solution.routes)):
            route1 = solution.routes[r1_idx]
            if len(route1) < 2:
                continue
            
            # Try chains of length 2 and 3
            for chain_len in [2, 3]:
                if len(route1) < chain_len:
                    continue
                
                for start_idx in range(len(route1) - chain_len + 1):
                    chain = route1[start_idx:start_idx + chain_len]
                    chain_demand = sum(self.instance.customers[cid].demand for cid in chain)
                    
                    # Try inserting chain in all other positions
                    for r2_idx in range(len(solution.routes)):
                        route2 = solution.routes[r2_idx]
                        route2_demand = solution.get_route_demand(route2)
                        
                        if r1_idx == r2_idx or route2_demand + chain_demand > self.instance.capacity:
                            continue
                        
                        for pos in range(len(route2) + 1):
                            new_solution = solution.copy()
                            # Remove chain from route1
                            new_solution.routes[r1_idx] = route1[:start_idx] + route1[start_idx+chain_len:]
                            # Insert chain in route2
                            new_solution.routes[r2_idx] = route2[:pos] + chain + route2[pos:]
                            
                            # Remove empty routes
                            new_solution.routes = [r for r in new_solution.routes if r]
                            
                            new_solution.calculate_distance()
                            
                            if new_solution.total_distance < best_solution.total_distance:
                                best_solution = new_solution
                                improved = True
        
        if improved:
            solution.routes = best_solution.routes
            solution.total_distance = best_solution.total_distance
        return improved
    
    def local_search(self, solution: Solution) -> Solution:
        """Apply local search until no improvement"""
        improved = True
        while improved:
            improved = False
            
            # Try 2-opt on each route
            for route_idx in range(len(solution.routes)):
                if self.two_opt(solution, route_idx):
                    improved = True
            
            # Try relocate
            if self.relocate(solution):
                improved = True
            
            # Try swap
            if self.swap(solution):
                improved = True
            
            # Try or-opt
            if self.or_opt(solution):
                improved = True
        
        return solution
    
    def shake(self, solution: Solution, k: int) -> Solution:
        """Shaking procedure: apply k random moves"""
        shaken = solution.copy()
        
        for _ in range(k):
            if len(shaken.routes) < 2:
                break
            
            # Random relocate move
            r1_idx = random.randint(0, len(shaken.routes) - 1)
            if not shaken.routes[r1_idx]:
                continue
            
            c1_idx = random.randint(0, len(shaken.routes[r1_idx]) - 1)
            customer_id = shaken.routes[r1_idx][c1_idx]
            customer = self.instance.customers[customer_id]
            
            # Find a valid route to insert
            valid_routes = []
            for r2_idx in range(len(shaken.routes)):
                if r1_idx == r2_idx:
                    continue
                route2_demand = shaken.get_route_demand(shaken.routes[r2_idx])
                if route2_demand + customer.demand <= self.instance.capacity:
                    valid_routes.append(r2_idx)
            
            if valid_routes:
                r2_idx = random.choice(valid_routes)
                pos = random.randint(0, len(shaken.routes[r2_idx]))
                
                # Perform relocate
                shaken.routes[r1_idx] = shaken.routes[r1_idx][:c1_idx] + shaken.routes[r1_idx][c1_idx+1:]
                shaken.routes[r2_idx] = shaken.routes[r2_idx][:pos] + [customer_id] + shaken.routes[r2_idx][pos:]
                
                # Remove empty routes
                shaken.routes = [r for r in shaken.routes if r]
        
        shaken.calculate_distance()
        return shaken
    
    def solve(self) -> Solution:
        """Main VNS algorithm"""
        start_time = time.time()
        
        # Generate initial solution
        if self.verbose:
            print("  Generating initial solution...", end=" ", flush=True)
        init_start = time.time()
        current = self.initial_solution()
        current = self.local_search(current)
        init_time = time.time() - init_start
        best = current.copy()
        
        if self.verbose:
            print(f"✓ Initial distance: {best.total_distance:.2f} ({init_time:.1f}s)")
        
        k_max = 5  # Maximum neighborhood size
        no_improvement = 0
        last_report_time = start_time
        report_interval = 10.0  # Report every 10 seconds
        last_improvement_time = start_time
        last_improvement_iteration = 0
        recent_iterations = []  # Track recent iteration times for better ETA
        recent_times = []
        
        for iteration in range(self.max_iterations):
            # Check time limit
            elapsed_time = time.time() - start_time
            if self.time_limit and elapsed_time > self.time_limit:
                if self.verbose:
                    print(f"\n  ⏱ Time limit reached ({self.time_limit:.0f}s)")
                break
            
            # Progress reporting with time estimation
            if self.verbose and (time.time() - last_report_time) >= report_interval:
                elapsed = time.time() - start_time
                progress = (iteration / self.max_iterations) * 100 if self.max_iterations > 0 else 0
                
                # Estimate remaining time
                if iteration > 0:
                    # Store recent iteration data for moving average
                    recent_iterations.append(iteration)
                    recent_times.append(elapsed)
                    # Keep only last 5 data points
                    if len(recent_iterations) > 5:
                        recent_iterations.pop(0)
                        recent_times.pop(0)
                    
                    # Calculate recent iteration rate (more accurate than overall)
                    if len(recent_iterations) >= 2:
                        recent_iter_diff = recent_iterations[-1] - recent_iterations[0]
                        recent_time_diff = recent_times[-1] - recent_times[0]
                        recent_rate = recent_iter_diff / recent_time_diff if recent_time_diff > 0 else 0
                    else:
                        recent_rate = iteration / elapsed if elapsed > 0 else 0
                    
                    # Overall rate as fallback
                    iterations_per_second = iteration / elapsed if elapsed > 0 else 0
                    remaining_iterations = self.max_iterations - iteration
                    
                    # Use recent rate if available, otherwise overall rate
                    rate_to_use = recent_rate if recent_rate > 0 else iterations_per_second
                    
                    # Estimate based on iterations
                    if rate_to_use > 0:
                        eta_iterations = remaining_iterations / rate_to_use
                    else:
                        eta_iterations = float('inf')
                    
                    # Estimate based on convergence (no improvement threshold)
                    iterations_since_improvement = iteration - last_improvement_iteration
                    if iterations_since_improvement > 0 and no_improvement > 0 and rate_to_use > 0:
                        # Estimate iterations until max_no_improvement is reached
                        remaining_no_improvement = self.max_no_improvement - no_improvement
                        if remaining_no_improvement > 0:
                            # Estimate iterations needed to reach threshold
                            iterations_to_threshold = remaining_no_improvement
                            # Estimate time based on current iteration rate
                            eta_convergence = iterations_to_threshold / rate_to_use
                        else:
                            eta_convergence = 0  # Already at threshold
                    else:
                        eta_convergence = float('inf')
                    
                    # If making regular improvements, algorithm likely to continue
                    # If not improving, likely to hit convergence threshold soon
                    # Use convergence estimate if it's reasonable, otherwise use iteration estimate
                    if eta_convergence < float('inf') and eta_convergence < eta_iterations:
                        # Likely to converge before max iterations
                        eta_seconds = eta_convergence
                        use_convergence_estimate = True
                    else:
                        # Will likely hit max iterations
                        eta_seconds = eta_iterations
                        use_convergence_estimate = False
                    
                    if eta_seconds < float('inf') and eta_seconds > 0:
                        eta_minutes = int(eta_seconds // 60)
                        eta_secs = int(eta_seconds % 60)
                        if eta_minutes > 60:
                            eta_hours = eta_minutes // 60
                            eta_mins = eta_minutes % 60
                            eta_str = f"ETA: ~{eta_hours}h {eta_mins}m"
                        elif eta_minutes > 0:
                            eta_str = f"ETA: ~{eta_minutes}m {eta_secs}s"
                        else:
                            eta_str = f"ETA: ~{eta_secs}s"
                        
                        # Add convergence warning if close to threshold
                        if no_improvement > self.max_no_improvement * 0.7:
                            eta_str += " (may stop early)"
                        elif use_convergence_estimate:
                            eta_str += " (est. convergence)"
                        # If convergence estimate is much less than iteration estimate, note it
                        elif eta_convergence < float('inf') and eta_convergence < eta_iterations * 0.3:
                            eta_str += " (likely to converge)"
                    else:
                        eta_str = "ETA: calculating..."
                    
                    print(f"  Iteration {iteration}/{self.max_iterations} ({progress:.1f}%) | "
                          f"Best: {best.total_distance:.2f} | Time: {elapsed:.1f}s | {eta_str}", flush=True)
                else:
                    print(f"  Iteration {iteration}/{self.max_iterations} ({progress:.1f}%) | "
                          f"Best: {best.total_distance:.2f} | Time: {elapsed:.1f}s", flush=True)
                
                last_report_time = time.time()
            
            k = 1
            
            while k <= k_max:
                # Shaking
                candidate = self.shake(current, k)
                
                # Local search
                candidate = self.local_search(candidate)
                
                # Acceptance criterion
                if candidate.total_distance < current.total_distance:
                    current = candidate
                    k = 1  # Return to first neighborhood
                    
                    if candidate.total_distance < best.total_distance:
                        best = candidate
                        no_improvement = 0
                        last_improvement_time = time.time()
                        last_improvement_iteration = iteration
                        if self.verbose:
                            elapsed = time.time() - start_time
                            print(f"  ✓ Improvement at iteration {iteration}: "
                                  f"{best.total_distance:.2f} ({elapsed:.1f}s)", flush=True)
                    else:
                        no_improvement += 1
                else:
                    k += 1
                
                if no_improvement >= self.max_no_improvement:
                    break
            
            if no_improvement >= self.max_no_improvement:
                if self.verbose:
                    print(f"  ⏹ No improvement for {self.max_no_improvement} iterations. Stopping.")
                break
        
        total_time = time.time() - start_time
        if self.verbose:
            print(f"  ✓ Completed in {total_time:.1f}s | Final distance: {best.total_distance:.2f}")
        
        return best

