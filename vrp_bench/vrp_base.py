import math
import random
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Set, Union
from travel_time_generator import sample_travel_time, get_distances


class VRPSolverBase(ABC):
    """Base class for all VRP solvers with proper time window and appear time handling"""
    
    def __init__(self, data: Dict):
        """Initialize with problem data and pre-compute optimizations"""
        # Handle data conversion
        self.locations = self._convert_inhomogeneous_array(data['locations'])
        self.demands = self._convert_inhomogeneous_array(data['demands'])
        self.num_vehicles = self._convert_inhomogeneous_array(data['num_vehicles'])
        self.vehicle_capacities = self._convert_inhomogeneous_array(data['vehicle_capacities'])
        
        # Handle appear times - these are CRITICAL for feasibility!
        self.appear_times = data.get('appear_times', None)
        if self.appear_times is not None:
            self.appear_times = self._convert_inhomogeneous_array(self.appear_times)
        
        # Handle time windows and time matrices for TWCVRP
        self.time_windows = data.get('time_windows', None)
        self.time_matrix = data.get('time_matrix', None)
        
        if self.time_windows is not None:
            self.time_windows = self._convert_inhomogeneous_array(self.time_windows)
            
        if self.time_matrix is not None:
            self.time_matrix = self._convert_inhomogeneous_array(self.time_matrix)
        
        # Validate and get dimensions
        self.num_instances = self._get_num_instances()
        self.num_nodes = self._get_num_nodes()
        
        # Pre-calculate distance matrices and other optimizations
        self._precompute_distances()
        self._precompute_depot_customer_indices()
        self._precompute_time_windows()
        self._precompute_appear_times()
    
    def _convert_inhomogeneous_array(self, data: Union[np.ndarray, list, tuple, int, float]) -> Union[List, np.ndarray]:
        """Convert inhomogeneous arrays to a format we can work with"""
        try:
            # If already a numpy array, check if it's homogeneous
            if isinstance(data, np.ndarray):
                # Check if it's an object array with lists of different lengths
                if data.dtype == object:
                    # Convert each element to numpy array if possible
                    converted_data = []
                    for item in data:
                        if isinstance(item, (list, tuple, np.ndarray)):
                            try:
                                converted_data.append(np.array(item, dtype=np.float64))
                            except:
                                converted_data.append(item)
                        else:
                            converted_data.append(item)
                    return converted_data
                else:
                    # Try standard conversion
                    try:
                        return data.astype(np.float64)
                    except:
                        return data
            
            # Handle scalar values
            if isinstance(data, (int, float)):
                return np.array([data], dtype=np.float64)
            
            # Handle list or tuple
            if isinstance(data, (list, tuple)):
                # Check if all elements have the same length
                if len(data) > 0:
                    first_len = len(data[0]) if hasattr(data[0], '__len__') else 1
                    all_same_length = all(
                        (len(item) if hasattr(item, '__len__') else 1) == first_len 
                        for item in data
                    )
                    
                    if all_same_length:
                        # Homogeneous - can convert to numpy array
                        return np.array(data, dtype=np.float64)
                    else:
                        # Inhomogeneous - keep as list
                        return [np.array(item, dtype=np.float64) if hasattr(item, '__len__') else np.array([item], dtype=np.float64) 
                                for item in data]
                else:
                    return np.array([], dtype=np.float64)
            
            # Fallback
            return data
            
        except Exception as e:
            return data
    
    def _get_num_instances(self) -> int:
        """Determine the number of instances from the data"""
        try:
            if isinstance(self.locations, list):
                return len(self.locations)
            elif hasattr(self.locations, 'shape'):
                if len(self.locations.shape) >= 2:
                    return self.locations.shape[0]
                elif len(self.locations.shape) == 1:
                    return 1
            return 1
        except:
            return 1
    
    def _get_num_nodes(self) -> int:
        """Determine the number of nodes per instance"""
        try:
            if isinstance(self.locations, list) and len(self.locations) > 0:
                if isinstance(self.locations[0], np.ndarray):
                    return self.locations[0].shape[0]
                elif hasattr(self.locations[0], '__len__'):
                    return len(self.locations[0]) // 2  # Assuming 2D coordinates
                else:
                    return 0
            elif hasattr(self.locations, 'shape'):
                if len(self.locations.shape) == 3:
                    return self.locations.shape[1]
                elif len(self.locations.shape) == 2:
                    return self.locations.shape[0]
            
            # Try to infer from demands
            if isinstance(self.demands, list) and len(self.demands) > 0:
                if isinstance(self.demands[0], np.ndarray):
                    return len(self.demands[0])
                elif hasattr(self.demands[0], '__len__'):
                    return len(self.demands[0])
            elif hasattr(self.demands, 'shape'):
                return self.demands.shape[-1]
            
            return 0
        except:
            return 0
    
    def _precompute_distances(self):
        """Pre-calculate distance matrices for all instances"""
        self.dist_matrices = []
        self.distance_dicts = []
        
        for i in range(self.num_instances):
            try:
                # Get locations for this instance
                if isinstance(self.locations, list):
                    locations = self.locations[i] if i < len(self.locations) else None
                else:
                    if len(self.locations.shape) == 3:
                        locations = self.locations[i]
                    elif len(self.locations.shape) == 2:
                        locations = self.locations if self.num_instances == 1 else self.locations[i]
                    else:
                        locations = None
                
                if locations is None:
                    raise ValueError(f"Could not extract locations for instance {i}")
                
                # Ensure locations is a 2D numpy array
                locations = np.array(locations, dtype=np.float64)
                if len(locations.shape) == 1:
                    # Assume 2D coordinates
                    if len(locations) % 2 == 0:
                        locations = locations.reshape(-1, 2)
                    else:
                        raise ValueError(f"Invalid location shape for instance {i}")
                
                # Calculate all pairwise distances
                num_nodes = len(locations)
                dist_matrix = np.zeros((num_nodes, num_nodes))
                distance_dict = {}
                
                for j in range(num_nodes):
                    for k in range(j+1, num_nodes):
                        dist = np.sqrt(np.sum((locations[j] - locations[k])**2))
                        dist_matrix[j, k] = dist
                        dist_matrix[k, j] = dist
                        distance_dict[(j, k)] = dist
                        distance_dict[(k, j)] = dist
                
                self.dist_matrices.append(dist_matrix)
                self.distance_dicts.append(distance_dict)
                
            except Exception as e:
                # Create empty distance matrix as fallback
                fallback_size = 10
                empty_dist = np.zeros((fallback_size, fallback_size))
                self.dist_matrices.append(empty_dist)
                self.distance_dicts.append({})
    
    def _precompute_depot_customer_indices(self):
        """Pre-compute depot and customer indices for all instances"""
        self.depot_indices = []
        self.customer_indices = []
        
        for i in range(self.num_instances):
            try:
                # Get demands for this instance
                if isinstance(self.demands, list):
                    demands = self.demands[i] if i < len(self.demands) else None
                else:
                    if len(self.demands.shape) == 2:
                        demands = self.demands[i]
                    elif len(self.demands.shape) == 1:
                        demands = self.demands
                    else:
                        demands = None
                
                if demands is None:
                    raise ValueError(f"Could not extract demands for instance {i}")
                
                # Ensure demands is 1D and proper dtype
                demands = np.array(demands, dtype=np.float64).flatten()
                
                # Find depots and customers
                depots = np.where(demands == 0)[0]
                customers = np.where(demands > 0)[0]
                
                self.depot_indices.append(depots)
                self.customer_indices.append(customers)
                
            except Exception as e:
                # Create fallback indices
                self.depot_indices.append(np.array([0], dtype=np.int64))  # Default to first node as depot
                self.customer_indices.append(np.array([], dtype=np.int64))
    
    def _precompute_time_windows(self):
        """Pre-compute time windows for efficient lookup"""
        self.time_window_dicts = []
        
        for i in range(self.num_instances):
            try:
                # Get time windows for this instance
                if self.time_windows is not None:
                    if isinstance(self.time_windows, list):
                        windows = self.time_windows[i] if i < len(self.time_windows) else None
                    else:
                        if len(self.time_windows.shape) == 3:
                            windows = self.time_windows[i]
                        elif len(self.time_windows.shape) == 2:
                            windows = self.time_windows
                        else:
                            windows = None
                    
                    if windows is not None:
                        # Convert to dictionary for fast lookup
                        windows_dict = {}
                        windows = np.array(windows, dtype=np.float64)
                        
                        for node_idx in range(len(windows)):
                            start_time = windows[node_idx][0]
                            end_time = windows[node_idx][1]
                            windows_dict[node_idx] = (start_time, end_time)
                        
                        self.time_window_dicts.append(windows_dict)
                    else:
                        self.time_window_dicts.append({})
                else:
                    self.time_window_dicts.append({})
                    
            except Exception as e:
                self.time_window_dicts.append({})
    
    def _precompute_appear_times(self):
        """Pre-compute appear times for efficient lookup"""
        self.appear_time_dicts = []
        
        for i in range(self.num_instances):
            try:
                # Get appear times for this instance
                if self.appear_times is not None:
                    if isinstance(self.appear_times, list):
                        appear_times = self.appear_times[i] if i < len(self.appear_times) else None
                    else:
                        if len(self.appear_times.shape) == 2:
                            appear_times = self.appear_times[i]
                        elif len(self.appear_times.shape) == 1:
                            appear_times = self.appear_times
                        else:
                            appear_times = None
                    
                    if appear_times is not None:
                        # Convert to dictionary for fast lookup
                        appear_dict = {}
                        appear_times = np.array(appear_times, dtype=np.float64)
                        
                        for node_idx in range(len(appear_times)):
                            appear_dict[node_idx] = appear_times[node_idx]
                        
                        self.appear_time_dicts.append(appear_dict)
                    else:
                        self.appear_time_dicts.append({})
                else:
                    self.appear_time_dicts.append({})
                    
            except Exception as e:
                self.appear_time_dicts.append({})
    
    @abstractmethod
    def solve_instance(self, instance_idx: int, num_realizations: int = 3) -> Dict:
        """Solve a single instance - must be implemented by subclasses"""
        pass
    
    def solve_all_instances(self, num_realizations: int = 3) -> Tuple[Dict, List[Dict]]:
        """Solve all instances"""
        all_results = []
        
        for instance_idx in range(self.num_instances):
            try:
                result = self.solve_instance(instance_idx, num_realizations)
                all_results.append(result)
            except Exception as e:
                # Add a default result to maintain consistency
                default_result = {
                    'total_cost': 0,
                    'waiting_time': 0,
                    'cvr': 100.0,  # High CVR indicates failure
                    'feasibility': 0,
                    'runtime': 0.001,
                    'robustness': 0
                }
                all_results.append(default_result)
        
        # Aggregate results
        if all_results:
            total_costs = [r['total_cost'] for r in all_results]
            waiting_times = [r.get('waiting_time', 0) for r in all_results]
            cvrs = [r['cvr'] for r in all_results]
            feasibilities = [r['feasibility'] for r in all_results]
            runtimes = [r['runtime'] for r in all_results]
            robustness = [r['robustness'] for r in all_results]
            
            avg_results = {
                'total_cost': np.mean(total_costs),
                'waiting_time': np.mean(waiting_times),
                'cvr': np.mean(cvrs),
                'feasibility': np.mean(feasibilities),
                'runtime': np.mean(runtimes),
                'robustness': np.mean(robustness)
            }
        else:
            # Return empty results if no instances were processed
            avg_results = {
                'total_cost': 0,
                'waiting_time': 0,
                'cvr': 100.0,
                'feasibility': 0,
                'runtime': 0,
                'robustness': 0
            }
        
        return avg_results, all_results
    
    # Common helper methods moved from solvers
    def _get_num_vehicles(self, instance_idx: int) -> int:
        """Get number of vehicles for instance"""
        if isinstance(self.num_vehicles, list):
            return self.num_vehicles[instance_idx] if instance_idx < len(self.num_vehicles) else 1
        elif hasattr(self.num_vehicles, 'shape'):
            if len(self.num_vehicles.shape) == 1:
                return int(self.num_vehicles[instance_idx]) if instance_idx < len(self.num_vehicles) else 1
            else:
                return int(self.num_vehicles)
        else:
            return 1
    
    def _get_demands(self, instance_idx: int) -> np.ndarray:
        """Get demands for instance"""
        if isinstance(self.demands, list):
            return self.demands[instance_idx] if instance_idx < len(self.demands) else np.array([])
        elif hasattr(self.demands, 'shape'):
            if len(self.demands.shape) == 2:
                return self.demands[instance_idx]
            else:
                return self.demands
        else:
            return np.array([])
    
    def _get_vehicle_capacities(self, instance_idx: int) -> np.ndarray:
        """Get vehicle capacities for instance"""
        if isinstance(self.vehicle_capacities, list):
            return self.vehicle_capacities[instance_idx] if instance_idx < len(self.vehicle_capacities) else np.array([100])
        elif hasattr(self.vehicle_capacities, 'shape'):
            if len(self.vehicle_capacities.shape) == 2:
                return self.vehicle_capacities[instance_idx]
            else:
                return self.vehicle_capacities
        else:
            return np.array([100])
    
    def _check_feasibility(self, routes: List[List[int]], instance_idx: int) -> Tuple[float, float, int]:
        """
        Fixed feasibility check with proper time window and appear time handling
        
        Returns:
            cvr: Constraint Violation Rate as percentage (0-100)
            feasibility: Binary feasibility (0 if any violations, 1 if feasible)
            tw_violations: Number of time window violations
        """
        total_violations = 0
        tw_violations = 0
        
        # Get instance data
        demands = self._get_demands(instance_idx)
        capacities = self._get_vehicle_capacities(instance_idx)
        time_windows = self.get_time_windows(instance_idx)
        appear_times = self.get_appear_times(instance_idx)
        depots, customers = self.get_depots_and_customers(instance_idx)
        
        # Count total number of customers that should be served
        total_customers = len(customers)
        
        # Track customers served and violations
        customers_served = 0
        capacity_violations = 0
        appear_time_violations = 0
        customer_visit_count = {}
        
        # Process each route
        for route_idx, route in enumerate(routes):
            if len(route) <= 2:  # Only depot to depot
                continue
            
            # Get vehicle capacity
            vehicle_capacity = capacities[route_idx] if route_idx < len(capacities) else (capacities[0] if len(capacities) > 0 else 100)
            route_demand = 0
            current_time = 0
            
            # Process route
            for i, node in enumerate(route):
                # Check customer visits
                if node in customers:
                    customers_served += 1
                    customer_visit_count[node] = customer_visit_count.get(node, 0) + 1
                    
                    # Add demand
                    if node < len(demands):
                        route_demand += demands[node]
                    
                    # Check if customer has appeared
                    if node in appear_times:
                        if current_time < appear_times[node]:
                            appear_time_violations += 1
                
                # Update current time
                if i > 0:
                    prev_node = route[i-1]
                    # Use sample_travel_time for stochastic routing
                    travel_time = sample_travel_time(prev_node, node, self.distance_dicts[instance_idx], current_time)
                    current_time += travel_time
                
                # Check time windows if available
                if node in time_windows and node not in depots:
                    start_time, end_time = time_windows[node]
                    
                    # Normalize time
                    current_time = current_time % 1440
                    
                    # Check violation
                    if current_time > end_time:
                        tw_violations += 1
                    elif current_time < start_time:
                        current_time = start_time
            
            # Check capacity violation
            if route_demand > vehicle_capacity * 1.001:  # Small tolerance for floating point
                capacity_violations += 1
        
        # Count violations
        total_violations = capacity_violations + tw_violations + appear_time_violations
        
        # Multiple visits are violations
        for customer, count in customer_visit_count.items():
            if count > 1:
                total_violations += (count - 1)
        
        # Unserved customers are violations
        unserved_customers = total_customers - customers_served
        total_violations += unserved_customers
        
        # Calculate CVR (violations per customer)
        if total_customers > 0:
            cvr = (total_violations / total_customers) * 100
        else:
            cvr = 0.0 if total_violations == 0 else 100.0
        
        # Calculate feasibility (binary: 1 if no violations, 0 otherwise)
        feasibility = 1.0 if total_violations == 0 else 0.0
        
        return cvr, feasibility, tw_violations
    
    def calculate_solution_cost(self, routes: List[List[int]], instance_idx: int, 
                              num_realizations: int = 3) -> Dict:
        """Calculate comprehensive solution cost with fixed metrics"""
        # Calculate stochastic costs
        total_cost, waiting_time, robustness = self._calculate_stochastic_cost(
            routes, instance_idx, num_realizations
        )
        
        # Check feasibility with fixed calculation
        cvr, feasibility, tw_violations = self._check_feasibility(routes, instance_idx)
        
        return {
            'total_cost': total_cost,
            'waiting_time': waiting_time,
            'cvr': cvr,  # Now correctly calculated as percentage
            'feasibility': feasibility,  # Now correctly 0 or 1
            'robustness': robustness,
            'routes': routes,
            'time_window_violations': tw_violations
        }
    
    def _calculate_stochastic_cost(self, routes: List[List[int]], instance_idx: int, 
                                  num_realizations: int = 3) -> Tuple[float, float, float]:
        """Calculate cost with stochastic travel times"""
        total_costs = []
        total_waiting_times = []
        
        for _ in range(num_realizations):
            total_cost = 0
            total_waiting_time = 0
            
            for route in routes:
                route_cost, route_waiting = self._simulate_route_execution(route, instance_idx)
                total_cost += route_cost
                total_waiting_time += route_waiting
            
            total_costs.append(total_cost)
            total_waiting_times.append(total_waiting_time)
        
        # Calculate statistics
        avg_cost = sum(total_costs) / len(total_costs) if total_costs else 0
        avg_waiting = sum(total_waiting_times) / len(total_waiting_times) if total_waiting_times else 0
        
        # Calculate robustness (variance)
        if len(total_costs) > 1:
            mean_cost = avg_cost
            variance = sum((cost - mean_cost) ** 2 for cost in total_costs) / len(total_costs)
            robustness = variance ** 0.5  # Standard deviation
        else:
            robustness = 0
        
        return avg_cost, avg_waiting, robustness
    
    def _simulate_route_execution(self, route: List[int], instance_idx: int) -> Tuple[float, float]:
        """Simulate route execution with stochastic travel times and time constraints"""
        if len(route) <= 1:
            return 0, 0
        
        current_time = 0
        total_cost = 0
        total_waiting = 0
        
        # Get time constraints
        time_windows = self.get_time_windows(instance_idx)
        appear_times = self.get_appear_times(instance_idx)
        distance_dict = self.distance_dicts[instance_idx] if instance_idx < len(self.distance_dicts) else {}
        
        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]
            
            # Get stochastic travel time
            travel_time = sample_travel_time(current_node, next_node, distance_dict, current_time)
            current_time += travel_time
            total_cost += travel_time
            
            # Check if customer has appeared
            if next_node in appear_times:
                if current_time < appear_times[next_node]:
                    waiting_time = appear_times[next_node] - current_time
                    total_waiting += waiting_time
                    current_time = appear_times[next_node]
            
            # Check time windows
            if next_node in time_windows:
                start_time, end_time = time_windows[next_node]
                
                # Wait if arriving before window opens
                if current_time < start_time:
                    waiting_time = start_time - current_time
                    total_waiting += waiting_time
                    current_time = start_time
        
        return total_cost, total_waiting
    
    def get_time_matrix(self, instance_idx: int) -> np.ndarray:
        """Get time matrix for a specific instance"""
        if self.time_matrix is None:
            # Fallback to distance matrix if no time matrix provided
            if instance_idx < len(self.dist_matrices):
                return self.dist_matrices[instance_idx]
            else:
                return np.array([[0]])
        
        try:
            if isinstance(self.time_matrix, list):
                time_mat = self.time_matrix[instance_idx] if instance_idx < len(self.time_matrix) else None
            else:
                if len(self.time_matrix.shape) == 3:
                    time_mat = self.time_matrix[instance_idx]
                elif len(self.time_matrix.shape) == 2:
                    time_mat = self.time_matrix
                else:
                    time_mat = None
            
            if time_mat is None:
                # Fallback to distance matrix
                if instance_idx < len(self.dist_matrices):
                    return self.dist_matrices[instance_idx]
                else:
                    return np.array([[0]])
            
            return np.array(time_mat, dtype=np.float64)
            
        except Exception as e:
            # Fallback to distance matrix
            if instance_idx < len(self.dist_matrices):
                return self.dist_matrices[instance_idx]
            else:
                return np.array([[0]])
    
    def get_time_windows(self, instance_idx: int) -> Dict[int, Tuple[float, float]]:
        """Get time windows for a specific instance"""
        if instance_idx < len(self.time_window_dicts):
            return self.time_window_dicts[instance_idx]
        else:
            return {}
    
    def get_appear_times(self, instance_idx: int) -> Dict[int, float]:
        """Get appear times for a specific instance"""
        if instance_idx < len(self.appear_time_dicts):
            return self.appear_time_dicts[instance_idx]
        else:
            return {}
    
    def get_depots_and_customers(self, instance_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fast lookup of pre-computed depot and customer nodes"""
        if instance_idx >= len(self.depot_indices):
            return np.array([0], dtype=np.int64), np.array([], dtype=np.int64)
        return self.depot_indices[instance_idx], self.customer_indices[instance_idx]
    
    def _create_empty_result(self) -> Dict:
        """Create empty result for failed instances"""
        return {
            'total_cost': 0,
            'waiting_time': 0,
            'cvr': 100.0,  # High CVR indicates complete failure
            'feasibility': 0.0,  # Not feasible
            'runtime': 0.001,
            'robustness': 0,
            'routes': [],
            'time_window_violations': 0
        }
    
    def _calculate_route_distance(self, route: List[int], dist_matrix: np.ndarray) -> float:
        """Calculate total distance of a route"""
        if len(route) <= 1:
            return 0
        
        total_distance = 0
        for i in range(len(route) - 1):
            try:
                total_distance += dist_matrix[route[i], route[i + 1]]
            except IndexError:
                # Skip invalid indices
                continue
        
        return total_distance