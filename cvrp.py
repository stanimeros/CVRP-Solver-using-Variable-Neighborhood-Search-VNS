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
    
    def _distance(self, cid1: int, cid2: int) -> float:
        """Calculate distance between two customer IDs (O(1))"""
        if cid1 == -1:  # Depot
            return self.instance.distance(self.instance.depot, self.instance.customers[cid2])
        elif cid2 == -1:  # Depot
            return self.instance.distance(self.instance.customers[cid1], self.instance.depot)
        else:
            return self.instance.distance(self.instance.customers[cid1], self.instance.customers[cid2])
    
    def _route_distance(self, route: List[int]) -> float:
        """Calculate total distance of a route including depot connections (O(n))"""
        if not route:
            return 0.0
        dist = self._distance(-1, route[0])  # Depot to first
        for i in range(len(route) - 1):
            dist += self._distance(route[i], route[i + 1])
        dist += self._distance(route[-1], -1)  # Last to depot
        return dist
    
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
        """2-opt neighborhood: reverse a segment of a route (delta evaluation)"""
        route = solution.routes[route_idx]
        if len(route) < 2:
            return False
        
        best_gain = 0.0
        best_i = -1
        best_j = -1
        
        for i in range(len(route)):
            for j in range(i + 2, len(route)):
                # Calculate delta: removed edges - added edges
                # Removed: (i, i+1) and (j, j+1)
                # Added: (i, j) and (i+1, j+1)
                removed_dist = self._distance(route[i], route[i+1])
                if j + 1 < len(route):
                    removed_dist += self._distance(route[j], route[j+1])
                else:
                    removed_dist += self._distance(route[j], -1)  # Last to depot
                
                added_dist = self._distance(route[i], route[j])
                if j + 1 < len(route):
                    added_dist += self._distance(route[i+1], route[j+1])
                else:
                    added_dist += self._distance(route[i+1], -1)  # Last to depot
                
                gain = removed_dist - added_dist
                
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = j
        
        if best_gain > 0.0:
            # Apply the best move
            route[best_i+1:best_j+1] = route[best_i+1:best_j+1][::-1]
            solution.total_distance -= best_gain
            return True
        
        return False
    
    def relocate(self, solution: Solution) -> bool:
        """Relocate neighborhood: move a customer from one route to another (delta evaluation)"""
        best_gain = 0.0
        best_r1_idx = -1
        best_c1_idx = -1
        best_r2_idx = -1
        best_pos = -1
        
        for r1_idx in range(len(solution.routes)):
            route1 = solution.routes[r1_idx]
            for c1_idx in range(len(route1)):
                customer_id = route1[c1_idx]
                customer = self.instance.customers[customer_id]
                
                # Calculate removal cost from route1
                prev_cid = route1[c1_idx - 1] if c1_idx > 0 else -1  # Depot if first
                next_cid = route1[c1_idx + 1] if c1_idx + 1 < len(route1) else -1  # Depot if last
                removed_dist = self._distance(prev_cid, customer_id) + self._distance(customer_id, next_cid)
                added_dist = self._distance(prev_cid, next_cid)
                route1_gain = removed_dist - added_dist
                
                # Try inserting in all other positions
                for r2_idx in range(len(solution.routes)):
                    if r1_idx == r2_idx:
                        continue
                    
                    route2 = solution.routes[r2_idx]
                    route2_demand = solution.get_route_demand(route2)
                    
                    # Capacity check first (cheap)
                    if route2_demand + customer.demand > self.instance.capacity:
                        continue
                    
                    for pos in range(len(route2) + 1):
                        # Calculate insertion cost in route2
                        if pos == 0:
                            # Insert at beginning: remove depot->first, add depot->customer->first
                            if route2:
                                removed_dist2 = self._distance(-1, route2[0])
                                added_dist2 = self._distance(-1, customer_id) + self._distance(customer_id, route2[0])
                            else:
                                # Empty route: add depot->customer->depot
                                removed_dist2 = 0.0
                                added_dist2 = self._distance(-1, customer_id) + self._distance(customer_id, -1)
                        elif pos == len(route2):
                            # Insert at end: remove last->depot, add last->customer->depot
                            removed_dist2 = self._distance(route2[-1], -1)
                            added_dist2 = self._distance(route2[-1], customer_id) + self._distance(customer_id, -1)
                        else:
                            # Insert in middle: remove prev->next, add prev->customer->next
                            prev_cid2 = route2[pos - 1]
                            next_cid2 = route2[pos]
                            removed_dist2 = self._distance(prev_cid2, next_cid2)
                            added_dist2 = self._distance(prev_cid2, customer_id) + self._distance(customer_id, next_cid2)
                        
                        route2_cost = added_dist2 - removed_dist2
                        total_gain = route1_gain - route2_cost
                        
                        if total_gain > best_gain:
                            best_gain = total_gain
                            best_r1_idx = r1_idx
                            best_c1_idx = c1_idx
                            best_r2_idx = r2_idx
                            best_pos = pos
        
        if best_gain > 0.0:
            # Apply the best move
            customer_id = solution.routes[best_r1_idx][best_c1_idx]
            
            # Remove from route1
            solution.routes[best_r1_idx].pop(best_c1_idx)
            
            # Insert in route2
            solution.routes[best_r2_idx].insert(best_pos, customer_id)
            
            # Remove empty routes
            solution.routes = [r for r in solution.routes if r]
            
            # Update distance incrementally
            solution.total_distance -= best_gain
            return True
        
        return False
    
    def swap(self, solution: Solution) -> bool:
        """Swap neighborhood: swap two customers between routes (delta evaluation)"""
        best_gain = 0.0
        best_r1_idx = -1
        best_c1_idx = -1
        best_r2_idx = -1
        best_c2_idx = -1
        
        for r1_idx in range(len(solution.routes)):
            route1 = solution.routes[r1_idx]
            for c1_idx in range(len(route1)):
                customer1_id = route1[c1_idx]
                customer1 = self.instance.customers[customer1_id]
                
                # Calculate edges around customer1 in route1
                prev1_cid = route1[c1_idx - 1] if c1_idx > 0 else -1
                next1_cid = route1[c1_idx + 1] if c1_idx + 1 < len(route1) else -1
                
                for r2_idx in range(r1_idx + 1, len(solution.routes)):
                    route2 = solution.routes[r2_idx]
                    route1_demand = solution.get_route_demand(route1) - customer1.demand
                    route2_demand = solution.get_route_demand(route2)
                    
                    for c2_idx in range(len(route2)):
                        customer2_id = route2[c2_idx]
                        customer2 = self.instance.customers[customer2_id]
                        
                        # Check capacity constraints first (cheap)
                        if (route1_demand + customer2.demand > self.instance.capacity or
                            route2_demand - customer2.demand + customer1.demand > self.instance.capacity):
                            continue
                        
                        # Calculate edges around customer2 in route2
                        prev2_cid = route2[c2_idx - 1] if c2_idx > 0 else -1
                        next2_cid = route2[c2_idx + 1] if c2_idx + 1 < len(route2) else -1
                        
                        # Calculate delta for route1: remove customer1, add customer2
                        removed1 = self._distance(prev1_cid, customer1_id) + self._distance(customer1_id, next1_cid)
                        added1 = self._distance(prev1_cid, customer2_id) + self._distance(customer2_id, next1_cid)
                        route1_gain = removed1 - added1
                        
                        # Calculate delta for route2: remove customer2, add customer1
                        removed2 = self._distance(prev2_cid, customer2_id) + self._distance(customer2_id, next2_cid)
                        added2 = self._distance(prev2_cid, customer1_id) + self._distance(customer1_id, next2_cid)
                        route2_gain = removed2 - added2
                        
                        total_gain = route1_gain + route2_gain
                        
                        if total_gain > best_gain:
                            best_gain = total_gain
                            best_r1_idx = r1_idx
                            best_c1_idx = c1_idx
                            best_r2_idx = r2_idx
                            best_c2_idx = c2_idx
        
        if best_gain > 0.0:
            # Apply the best swap
            customer1_id = solution.routes[best_r1_idx][best_c1_idx]
            customer2_id = solution.routes[best_r2_idx][best_c2_idx]
            
            solution.routes[best_r1_idx][best_c1_idx] = customer2_id
            solution.routes[best_r2_idx][best_c2_idx] = customer1_id
            
            # Update distance incrementally
            solution.total_distance -= best_gain
            return True
        
        return False
    
    def or_opt(self, solution: Solution) -> bool:
        """Or-opt neighborhood: relocate a chain of 2-3 consecutive customers (delta evaluation)"""
        best_gain = 0.0
        best_r1_idx = -1
        best_start_idx = -1
        best_chain_len = -1
        best_r2_idx = -1
        best_pos = -1
        
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
                    
                    # Calculate removal cost from route1
                    prev_cid = route1[start_idx - 1] if start_idx > 0 else -1
                    next_cid = route1[start_idx + chain_len] if start_idx + chain_len < len(route1) else -1
                    chain_first = chain[0]
                    chain_last = chain[-1]
                    
                    removed_dist = self._distance(prev_cid, chain_first) + self._distance(chain_last, next_cid)
                    added_dist = self._distance(prev_cid, next_cid)
                    route1_gain = removed_dist - added_dist
                    
                    # Try inserting chain in all other positions
                    for r2_idx in range(len(solution.routes)):
                        route2 = solution.routes[r2_idx]
                        route2_demand = solution.get_route_demand(route2)
                        
                        if r1_idx == r2_idx or route2_demand + chain_demand > self.instance.capacity:
                            continue
                        
                        for pos in range(len(route2) + 1):
                            # Calculate insertion cost in route2
                            if pos == 0:
                                # Insert at beginning
                                if route2:
                                    removed_dist2 = self._distance(-1, route2[0])
                                    added_dist2 = self._distance(-1, chain_first) + self._distance(chain_last, route2[0])
                                else:
                                    # Empty route
                                    removed_dist2 = 0.0
                                    added_dist2 = self._distance(-1, chain_first) + self._distance(chain_last, -1)
                            elif pos == len(route2):
                                # Insert at end
                                removed_dist2 = self._distance(route2[-1], -1)
                                added_dist2 = self._distance(route2[-1], chain_first) + self._distance(chain_last, -1)
                            else:
                                # Insert in middle
                                prev_cid2 = route2[pos - 1]
                                next_cid2 = route2[pos]
                                removed_dist2 = self._distance(prev_cid2, next_cid2)
                                added_dist2 = self._distance(prev_cid2, chain_first) + self._distance(chain_last, next_cid2)
                            
                            route2_cost = added_dist2 - removed_dist2
                            total_gain = route1_gain - route2_cost
                            
                            if total_gain > best_gain:
                                best_gain = total_gain
                                best_r1_idx = r1_idx
                                best_start_idx = start_idx
                                best_chain_len = chain_len
                                best_r2_idx = r2_idx
                                best_pos = pos
        
        if best_gain > 0.0:
            # Apply the best move
            chain = solution.routes[best_r1_idx][best_start_idx:best_start_idx + best_chain_len]
            
            # Remove chain from route1
            solution.routes[best_r1_idx] = (solution.routes[best_r1_idx][:best_start_idx] + 
                                            solution.routes[best_r1_idx][best_start_idx + best_chain_len:])
            
            # Insert chain in route2
            solution.routes[best_r2_idx] = (solution.routes[best_r2_idx][:best_pos] + 
                                           chain + 
                                           solution.routes[best_r2_idx][best_pos:])
            
            # Remove empty routes
            solution.routes = [r for r in solution.routes if r]
            
            # Update distance incrementally
            solution.total_distance -= best_gain
            return True
        
        return False
    
    def two_opt_star(self, solution: Solution) -> bool:
        """2-opt* neighborhood: exchange tails of two different routes (delta evaluation)"""
        if len(solution.routes) < 2:
            return False
        
        best_gain = 0.0
        best_r1_idx = -1
        best_i = -1
        best_r2_idx = -1
        best_j = -1
        
        for r1_idx in range(len(solution.routes)):
            route1 = solution.routes[r1_idx]
            if len(route1) < 1:
                continue
            
            route1_demand = solution.get_route_demand(route1)
            
            for r2_idx in range(r1_idx + 1, len(solution.routes)):
                route2 = solution.routes[r2_idx]
                if len(route2) < 1:
                    continue
                
                route2_demand = solution.get_route_demand(route2)
                
                # Try all positions in route1
                for i in range(len(route1)):
                    # Try all positions in route2
                    for j in range(len(route2)):
                        # 2-opt*: exchange tails
                        # New route1: route1[0..i] + route2[j+1..end]
                        # New route2: route2[0..j] + route1[i+1..end]
                        
                        # Calculate demands for new routes
                        route1_head_demand = sum(self.instance.customers[route1[k]].demand for k in range(i + 1))
                        route1_tail_demand = route1_demand - route1_head_demand
                        route2_head_demand = sum(self.instance.customers[route2[k]].demand for k in range(j + 1))
                        route2_tail_demand = route2_demand - route2_head_demand
                        
                        # Check capacity constraints
                        new_route1_demand = route1_head_demand + route2_tail_demand
                        new_route2_demand = route2_head_demand + route1_tail_demand
                        
                        if (new_route1_demand > self.instance.capacity or 
                            new_route2_demand > self.instance.capacity):
                            continue
                        
                        # Get the nodes involved
                        node_i = route1[i]
                        node_i_next = route1[i + 1] if i + 1 < len(route1) else -1  # Depot if last
                        node_j = route2[j]
                        node_j_next = route2[j + 1] if j + 1 < len(route2) else -1  # Depot if last
                        
                        # Calculate removed edges
                        removed1 = self._distance(node_i, node_i_next)
                        removed2 = self._distance(node_j, node_j_next)
                        
                        # Calculate added edges (2-opt* connections)
                        added1 = self._distance(node_i, node_j_next)
                        added2 = self._distance(node_j, node_i_next)
                        
                        # Calculate gain
                        gain = (removed1 + removed2) - (added1 + added2)
                        
                        # Check if this creates better routes
                        if gain > best_gain:
                            best_gain = gain
                            best_r1_idx = r1_idx
                            best_i = i
                            best_r2_idx = r2_idx
                            best_j = j
        
        if best_gain > 0.0:
            # Apply the best 2-opt* move
            route1 = solution.routes[best_r1_idx]
            route2 = solution.routes[best_r2_idx]
            
            # Exchange tails: route1[0..i] + route2[j+1..end] and route2[0..j] + route1[i+1..end]
            new_route1 = route1[:best_i + 1] + route2[best_j + 1:]
            new_route2 = route2[:best_j + 1] + route1[best_i + 1:]
            
            # Update routes
            solution.routes[best_r1_idx] = new_route1
            solution.routes[best_r2_idx] = new_route2
            
            # Remove empty routes
            solution.routes = [r for r in solution.routes if r]
            
            # Update distance incrementally
            solution.total_distance -= best_gain
            return True
        
        return False
    
    def local_search(self, solution: Solution) -> Solution:
        """Apply local search until no improvement"""
        improved = True
        while improved:
            improved = False
            
            # Try 2-opt on each route
            for route_idx in range(len(solution.routes)):
                if self.two_opt(solution, route_idx):
                    improved = True
            
            # Try 2-opt* (inter-route)
            if self.two_opt_star(solution):
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
        """Ruin & Recreate shaking: remove k customers and reinsert using best insertion"""
        shaken = solution.copy()
        
        if len(shaken.routes) == 0:
            return shaken
        
        # Calculate number of customers to ruin (proportional to k)
        total_customers = sum(len(route) for route in shaken.routes)
        if total_customers == 0:
            return shaken
        
        # Ruin: remove k customers (or k% of customers, whichever is more meaningful)
        num_to_ruin = min(k * 3, total_customers // 3)  # Remove up to k*3 customers or 1/3 of total
        num_to_ruin = max(1, num_to_ruin)  # At least 1
        
        # Collect all customers with their route indices
        customers_to_ruin = []
        for r_idx, route in enumerate(shaken.routes):
            for c_idx, customer_id in enumerate(route):
                customers_to_ruin.append((r_idx, c_idx, customer_id))
        
        # Randomly select customers to ruin
        if len(customers_to_ruin) > num_to_ruin:
            ruined = random.sample(customers_to_ruin, num_to_ruin)
        else:
            ruined = customers_to_ruin.copy()
        
        # Sort by route index (descending) and position (descending) to avoid index shifting issues
        ruined.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        # Remove ruined customers and track them
        ruined_customers = []
        for r_idx, c_idx, customer_id in ruined:
            if r_idx < len(shaken.routes) and c_idx < len(shaken.routes[r_idx]):
                shaken.routes[r_idx].pop(c_idx)
                ruined_customers.append(customer_id)
        
        # Remove empty routes
        shaken.routes = [r for r in shaken.routes if r]
        
        # Recreate: Reinsert customers using Best Insertion heuristic
        random.shuffle(ruined_customers)  # Randomize insertion order
        
        for customer_id in ruined_customers:
            customer = self.instance.customers[customer_id]
            best_cost = float('inf')
            best_r_idx = -1
            best_pos = -1
            
            # Find best insertion position
            for r_idx in range(len(shaken.routes)):
                route = shaken.routes[r_idx]
                route_demand = shaken.get_route_demand(route)
                
                # Check capacity
                if route_demand + customer.demand > self.instance.capacity:
                    continue
                
                # Try all insertion positions
                for pos in range(len(route) + 1):
                    # Calculate insertion cost (delta)
                    if pos == 0:
                        if route:
                            cost = self._distance(-1, customer_id) + self._distance(customer_id, route[0]) - self._distance(-1, route[0])
                        else:
                            cost = self._distance(-1, customer_id) + self._distance(customer_id, -1)
                    elif pos == len(route):
                        cost = self._distance(route[-1], customer_id) + self._distance(customer_id, -1) - self._distance(route[-1], -1)
                    else:
                        prev_cid = route[pos - 1]
                        next_cid = route[pos]
                        cost = (self._distance(prev_cid, customer_id) + self._distance(customer_id, next_cid) - 
                               self._distance(prev_cid, next_cid))
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_r_idx = r_idx
                        best_pos = pos
            
            # Insert at best position (always insert, create new route if necessary)
            if best_r_idx >= 0:
                shaken.routes[best_r_idx].insert(best_pos, customer_id)
            else:
                # Create new route if no capacity available in existing routes
                shaken.routes.append([customer_id])
        
        # Recalculate distance
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

