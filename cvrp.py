"""
Capacitated Vehicle Routing Problem (CVRP) Solver using Variable Neighborhood Search (VNS)
"""

import math
import random
import copy
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
                 max_no_improvement: int = 100, random_seed: Optional[int] = None):
        self.instance = instance
        self.max_iterations = max_iterations
        self.max_no_improvement = max_no_improvement
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
        # Generate initial solution
        current = self.initial_solution()
        current = self.local_search(current)
        best = current.copy()
        
        k_max = 5  # Maximum neighborhood size
        no_improvement = 0
        
        for iteration in range(self.max_iterations):
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
                    else:
                        no_improvement += 1
                else:
                    k += 1
                
                if no_improvement >= self.max_no_improvement:
                    break
            
            if no_improvement >= self.max_no_improvement:
                break
        
        return best

