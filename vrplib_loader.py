"""
VRPLIB instance loader for CVRP problems
"""

import re
import os
from typing import List, Optional, Dict, Any
from cvrp import CVRPInstance, Customer

# Try to import vrplib package, fallback to custom parser
try:
    import vrplib
    VRPLIB_AVAILABLE = True
except ImportError:
    VRPLIB_AVAILABLE = False


def load_vrplib_instance(filepath: str) -> CVRPInstance:
    """
    Load a CVRP instance from VRPLIB format
    
    Uses vrplib package if available, otherwise falls back to custom parser.
    
    Format specification:
    - NAME: instance name
    - TYPE: CVRP
    - DIMENSION: number of nodes
    - CAPACITY: vehicle capacity
    - NODE_COORD_SECTION: coordinates
    - DEMAND_SECTION: customer demands
    - DEPOT_SECTION: depot node(s)
    """
    if VRPLIB_AVAILABLE:
        try:
            return _load_with_vrplib(filepath)
        except Exception as e:
            print(f"Warning: Failed to load with vrplib package: {e}")
            print("Falling back to custom parser...")
    
    return _load_custom_parser(filepath)


def _load_with_vrplib(filepath: str) -> CVRPInstance:
    """Load instance using vrplib package"""
    import numpy as np
    data = vrplib.read_instance(filepath)
    
    # Extract instance name
    name = os.path.splitext(os.path.basename(filepath))[0]
    if 'name' in data:
        name = data['name']
    elif 'NAME' in data:
        name = data['NAME']
    
    # Get capacity - try different possible keys
    capacity = 0.0
    if 'capacity' in data:
        cap_val = data['capacity']
        if isinstance(cap_val, (list, np.ndarray)) and len(cap_val) > 0:
            capacity = float(cap_val[0])
        else:
            capacity = float(cap_val)
    elif 'CAPACITY' in data:
        capacity = float(data['CAPACITY'])
    elif 'CAPACITY_SECTION' in data:
        # If multiple capacities, use the first one
        cap_section = data['CAPACITY_SECTION']
        if isinstance(cap_section, dict):
            capacity = float(list(cap_section.values())[0])
        elif isinstance(cap_section, (list, np.ndarray)) and len(cap_section) > 0:
            capacity = float(cap_section[0])
    
    # Get coordinates - vrplib returns numpy arrays
    coords_array = data.get('node_coord')
    if coords_array is None:
        coords_array = data.get('NODE_COORD_SECTION')
    if coords_array is None:
        raise ValueError("NODE_COORD_SECTION/node_coord not found")
    
    # Convert to dict if it's an array (1-indexed)
    if isinstance(coords_array, np.ndarray):
        coords = {}
        for i in range(len(coords_array)):
            node_id = i + 1  # 1-indexed
            if len(coords_array[i]) >= 2:
                coords[node_id] = (float(coords_array[i][0]), float(coords_array[i][1]))
    elif isinstance(coords_array, dict):
        coords = coords_array
    else:
        raise ValueError("Invalid coordinate format")
    
    # Get demands - vrplib returns numpy arrays
    demands_array = data.get('demand')
    if demands_array is None:
        demands_array = data.get('DEMAND_SECTION')
    if demands_array is None:
        raise ValueError("DEMAND_SECTION/demand not found")
    
    # Convert to dict if it's an array (1-indexed)
    if isinstance(demands_array, np.ndarray):
        demands = {}
        for i in range(len(demands_array)):
            node_id = i + 1  # 1-indexed
            demands[node_id] = float(demands_array[i])
    elif isinstance(demands_array, dict):
        demands = demands_array
    else:
        raise ValueError("Invalid demand format")
    
    # Get depot - try different formats
    depot_id = 1  # Default depot is usually node 1
    depot_array = data.get('depot')
    if depot_array is None:
        depot_array = data.get('DEPOT_SECTION')
    
    if depot_array is not None:
        if isinstance(depot_array, np.ndarray):
            if len(depot_array) > 0:
                depot_id = int(depot_array[0])
        elif isinstance(depot_array, list):
            if len(depot_array) > 0:
                depot_id = int(depot_array[0])
        elif isinstance(depot_array, dict):
            depot_id = int(list(depot_array.keys())[0])
        else:
            depot_id = int(depot_array)
    
    # Handle case where depot might not be in coords (use node 1)
    if depot_id not in coords:
        if 1 in coords:
            depot_id = 1
        elif len(coords) > 0:
            depot_id = min(coords.keys())
    
    # Create customers (excluding depot)
    customers = []
    for node_id in sorted(coords.keys()):
        if node_id == depot_id:
            continue
        
        coord = coords[node_id]
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            x, y = float(coord[0]), float(coord[1])
        else:
            continue
        
        demand = 0.0
        if node_id in demands:
            demand_val = demands[node_id]
            if isinstance(demand_val, (list, tuple, np.ndarray)):
                demand = float(demand_val[0])
            else:
                demand = float(demand_val)
        
        customer = Customer(id=len(customers), x=x, y=y, demand=demand)
        customers.append(customer)
    
    # Create depot
    depot_coord = coords[depot_id]
    if isinstance(depot_coord, (list, tuple)) and len(depot_coord) >= 2:
        depot_x, depot_y = float(depot_coord[0]), float(depot_coord[1])
    else:
        raise ValueError(f"Invalid depot coordinates for node {depot_id}")
    
    depot = Customer(id=-1, x=depot_x, y=depot_y, demand=0.0)
    
    dimension = len(customers) + 1
    if 'dimension' in data:
        dim_val = data['dimension']
        dimension = int(dim_val) if not isinstance(dim_val, (list, np.ndarray)) else int(dim_val[0])
    elif 'DIMENSION' in data:
        dimension = int(data['DIMENSION'])
    
    return CVRPInstance(
        name=name,
        dimension=dimension,
        capacity=capacity,
        depot=depot,
        customers=customers
    )


def _load_custom_parser(filepath: str) -> CVRPInstance:
    """Load instance using custom parser (fallback)"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse header
    name_match = re.search(r'NAME\s*:\s*(.+)', content)
    name = name_match.group(1).strip() if name_match else "Unknown"
    
    dimension_match = re.search(r'DIMENSION\s*:\s*(\d+)', content)
    dimension = int(dimension_match.group(1)) if dimension_match else 0
    
    capacity_match = re.search(r'CAPACITY\s*:\s*(\d+)', content)
    capacity = float(capacity_match.group(1)) if capacity_match else 0
    
    # Parse coordinates
    coord_section = re.search(r'NODE_COORD_SECTION\s*(.*?)(?=DEMAND_SECTION|DEPOT_SECTION|EOF)', content, re.DOTALL)
    if not coord_section:
        raise ValueError("NODE_COORD_SECTION not found")
    
    coordinates = {}
    for line in coord_section.group(1).strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('-1'):
            continue
        parts = line.split()
        if len(parts) >= 3:
            node_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            coordinates[node_id] = (x, y)
    
    # Parse demands
    demand_section = re.search(r'DEMAND_SECTION\s*(.*?)(?=DEPOT_SECTION|EOF)', content, re.DOTALL)
    if not demand_section:
        raise ValueError("DEMAND_SECTION not found")
    
    demands = {}
    for line in demand_section.group(1).strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('-1'):
            continue
        parts = line.split()
        if len(parts) >= 2:
            node_id = int(parts[0])
            demand = float(parts[1])
            demands[node_id] = demand
    
    # Parse depot
    depot_section = re.search(r'DEPOT_SECTION\s*(.*?)(?=EOF|$)', content, re.DOTALL)
    if not depot_section:
        raise ValueError("DEPOT_SECTION not found")
    
    depot_id = None
    for line in depot_section.group(1).strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('-1'):
            # Handle tab-separated values (take first number)
            parts = line.split()
            if parts:
                try:
                    depot_id = int(parts[0])
                    break
                except ValueError:
                    continue
    
    if depot_id is None:
        raise ValueError("Depot ID not found")
    
    # Create customers (excluding depot)
    customers = []
    for node_id in sorted(coordinates.keys()):
        if node_id == depot_id:
            continue
        x, y = coordinates[node_id]
        demand = demands.get(node_id, 0.0)
        # Map to 0-indexed
        customer = Customer(id=len(customers), x=x, y=y, demand=demand)
        customers.append(customer)
    
    # Create depot
    depot_x, depot_y = coordinates[depot_id]
    depot = Customer(id=-1, x=depot_x, y=depot_y, demand=0.0)
    
    return CVRPInstance(
        name=name,
        dimension=dimension,
        capacity=capacity,
        depot=depot,
        customers=customers
    )


def load_best_known_solution(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load best-known solution from .sol file
    
    Returns dictionary with:
    - 'distance': best-known distance
    - 'routes': list of routes (each route is list of customer IDs)
    """
    if not os.path.exists(filepath):
        return None
    
    try:
        if VRPLIB_AVAILABLE:
            return vrplib.read_solution(filepath)
        else:
            return _load_solution_custom(filepath)
    except Exception as e:
        print(f"Warning: Failed to load solution file: {e}")
        return None


def _load_solution_custom(filepath: str) -> Optional[Dict[str, Any]]:
    """Load solution using custom parser"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse route section
    route_section = re.search(r'Route\s*#\s*\d+\s*:\s*(.*?)(?=Route|Cost|$)', content, re.DOTALL | re.IGNORECASE)
    if not route_section:
        # Try alternative format
        routes = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('Route') or ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    route_str = parts[1].strip()
                    route = [int(x.strip()) for x in route_str.split() if x.strip().isdigit()]
                    if route:
                        routes.append(route)
        
        if routes:
            # Extract cost/distance
            cost_match = re.search(r'Cost\s*[:\-]?\s*(\d+\.?\d*)', content, re.IGNORECASE)
            distance = float(cost_match.group(1)) if cost_match else None
            
            return {
                'routes': routes,
                'cost': distance
            }
    
    return None


def create_simple_instance(name: str, customers_data: List[tuple], capacity: float, 
                          depot: tuple = (0, 0)) -> CVRPInstance:
    """
    Create a simple CVRP instance from data
    
    Args:
        name: Instance name
        customers_data: List of tuples (x, y, demand)
        capacity: Vehicle capacity
        depot: Depot coordinates (x, y)
    """
    depot_customer = Customer(id=-1, x=depot[0], y=depot[1], demand=0.0)
    customers = []
    
    for i, (x, y, demand) in enumerate(customers_data):
        customers.append(Customer(id=i, x=x, y=y, demand=demand))
    
    return CVRPInstance(
        name=name,
        dimension=len(customers) + 1,
        capacity=capacity,
        depot=depot_customer,
        customers=customers
    )

