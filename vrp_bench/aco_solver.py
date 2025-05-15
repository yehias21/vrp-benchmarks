import time
import random
import numpy as np
from typing import List, Dict, Tuple, Set
from vrp_base import VRPSolverBase
from travel_time_generator import sample_travel_time
from nn_2opt_solver import NN2optSolver


class ACOSolver(VRPSolverBase):
    """Simple and fast Ant Colony Optimization solver for VRP"""
    
    def __init__(self, data: Dict):
        """Initialize with problem data"""
        super().__init__(data)
        
        # ACO parameters (optimized for speed)
        self.num_ants = 3  # Very small number of ants for speed
        self.max_iterations = 5  # Limited iterations
        self.alpha = 1.0  # Pheromone importance
        self.beta = 2.0   # Heuristic importance
        self.rho = 0.5    # Evaporation rate
        self.Q = 100      # Pheromone constant
        
        # Pheromone matrices (will be initialized per instance)
        self.pheromone_matrices = []
        
        # Create initial solver for baseline
        self.nn_solver = NN2optSolver(data)
        self.nn_solver.debug = False
        
        # Debug flag
        self.debug = False
    
    def solve_instance(self, instance_idx: int, num_realizations: int = 3) -> Dict:
        """Solve using fast ACO"""
        start_time = time.time()
        
        try:
            # Get problem data
            num_vehicles = self._get_num_vehicles(instance_idx)
            depots, customers = self.get_depots_and_customers(instance_idx)
            
            if len(customers) == 0:
                return self._create_empty_result()
            
            # Initialize pheromone matrix
            num_nodes = len(depots) + len(customers)
            pheromone_matrix = np.ones((num_nodes, num_nodes))
            
            # Get initial solution from NN+2opt for comparison
            initial_result = self.nn_solver.solve_instance(instance_idx, 1)
            best_routes = initial_result['routes']
            best_cost = initial_result['total_cost']
            
            if self.debug:
                print(f"\n--- ACO Instance {instance_idx} ---")
                print(f"Initial cost: {best_cost:.1f}")
            
            # ACO main loop
            for iteration in range(self.max_iterations):
                # Construct solutions for all ants
                ant_routes = []
                ant_costs = []
                
                for ant in range(self.num_ants):
                    # Build solution for this ant
                    routes = self._construct_solution(instance_idx, pheromone_matrix, num_vehicles, depots, customers)
                    
                    # Calculate cost
                    cost = self._calculate_route_cost(routes, instance_idx)
                    
                    ant_routes.append(routes)
                    ant_costs.append(cost)
                    
                    # Update best solution
                    if cost < best_cost:
                        best_cost = cost
                        best_routes = routes
                
                # Update pheromone levels
                self._update_pheromones(pheromone_matrix, ant_routes, ant_costs)
                
                if self.debug:
                    print(f"Iteration {iteration}: best_cost={best_cost:.1f}")
            
            # Apply 2-opt improvement to best solution
            best_routes = self._apply_local_search(best_routes, instance_idx)
            
            # Calculate final solution cost
            result = self.calculate_solution_cost(best_routes, instance_idx, num_realizations)
            result['runtime'] = time.time() - start_time
            
            if self.debug:
                print(f"Final cost: {result['total_cost']:.1f}")
                print(f"Final CVR: {result['cvr']:.1f}%")
                print("------------------------")
            
            return result
            
        except Exception as e:
            print(f"ACO Error: {e}")
            import traceback
            traceback.print_exc()
            return self._create_empty_result()
    
    def _construct_solution(self, instance_idx: int, pheromone_matrix: np.ndarray, 
                          num_vehicles: int, depots: np.ndarray, customers: np.ndarray) -> List[List[int]]:
        """Construct solution using ACO probabilistic selection"""
        routes = []
        unvisited = set(customers)
        
        # Get instance data
        dist_matrix = self.dist_matrices[instance_idx] if instance_idx < len(self.dist_matrices) else None
        demands = self._get_demands(instance_idx)
        capacities = self._get_vehicle_capacities(instance_idx)
        time_windows = self.get_time_windows(instance_idx)
        appear_times = self.get_appear_times(instance_idx)
        distance_dict = self.distance_dicts[instance_idx] if instance_idx < len(self.distance_dicts) else {}
        
        if dist_matrix is None:
            return []
        
        # Precompute heuristic values (1/distance)
        eta = 1.0 / (dist_matrix + 1e-10)  # Avoid division by zero
        
        # Build routes for each vehicle
        for vehicle in range(num_vehicles):
            if not unvisited:
                break
            
            # Get vehicle data
            vehicle_capacity = capacities[vehicle] if vehicle < len(capacities) else (capacities[0] if len(capacities) > 0 else 100)
            start_depot = depots[vehicle] if vehicle < len(depots) else depots[0]
            
            route = [start_depot]
            current = start_depot
            route_demand = 0
            current_time = 0
            
            # Build route using ACO probabilistic selection
            while unvisited:
                # Calculate probabilities for feasible customers
                probabilities = {}
                total_prob = 0
                
                for customer in unvisited:
                    # Quick feasibility checks
                    if customer < len(demands):
                        customer_demand = demands[customer]
                        if route_demand + customer_demand > vehicle_capacity:
                            continue
                    
                    # Check time constraints (simplified)
                    travel_time = sample_travel_time(current, customer, distance_dict, current_time)
                    arrival_time = current_time + travel_time
                    
                    # Appear time check
                    if customer in appear_times and arrival_time < appear_times[customer]:
                        continue
                    
                    # Time window check
                    if customer in time_windows:
                        _, end_time = time_windows[customer]
                        if arrival_time > end_time:
                            continue
                    
                    # Calculate probability
                    if current < pheromone_matrix.shape[0] and customer < pheromone_matrix.shape[1]:
                        pheromone = pheromone_matrix[current, customer]
                        heuristic = eta[current, customer] if current < eta.shape[0] and customer < eta.shape[1] else 1.0
                        
                        prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
                        probabilities[customer] = prob
                        total_prob += prob
                
                # Select next customer
                if not probabilities:
                    break
                
                # Roulette wheel selection
                r = random.random() * total_prob
                cumulative_prob = 0
                selected_customer = None
                
                for customer, prob in probabilities.items():
                    cumulative_prob += prob
                    if cumulative_prob >= r:
                        selected_customer = customer
                        break
                
                if selected_customer is None:
                    break
                
                # Add customer to route
                route.append(selected_customer)
                unvisited.remove(selected_customer)
                route_demand += demands[selected_customer] if selected_customer < len(demands) else 0
                
                # Update time
                travel_time = sample_travel_time(current, selected_customer, distance_dict, current_time)
                current_time += travel_time
                
                # Wait if necessary
                if selected_customer in appear_times:
                    current_time = max(current_time, appear_times[selected_customer])
                
                if selected_customer in time_windows:
                    start_time, _ = time_windows[selected_customer]
                    current_time = max(current_time, start_time)
                
                current = selected_customer
            
            # Return to depot
            route.append(start_depot)
            routes.append(route)
        
        # Create single-customer routes for remaining unvisited customers
        while unvisited:
            start_depot = depots[0] if len(depots) > 0 else 0
            customer = unvisited.pop()
            routes.append([start_depot, customer, start_depot])
        
        return routes
    
    def _calculate_route_cost(self, routes: List[List[int]], instance_idx: int) -> float:
        """Calculate simple route cost"""
        total_cost = 0
        distance_dict = self.distance_dicts[instance_idx] if instance_idx < len(self.distance_dicts) else {}
        
        for route in routes:
            for i in range(len(route) - 1):
                if (route[i], route[i+1]) in distance_dict:
                    total_cost += distance_dict[(route[i], route[i+1])]
                else:
                    # Fallback to direct distance calculation
                    total_cost += np.sqrt(np.sum((self.locations[instance_idx][route[i]] - 
                                                self.locations[instance_idx][route[i+1]])**2))
        
        return total_cost
    
    def _update_pheromones(self, pheromone_matrix: np.ndarray, ant_routes: List[List[List[int]]], 
                          ant_costs: List[float]):
        """Update pheromone levels"""
        # Evaporation
        pheromone_matrix *= (1 - self.rho)
        
        # Add pheromone from ants
        for routes, cost in zip(ant_routes, ant_costs):
            if cost == 0:
                continue
            
            delta_pheromone = self.Q / cost
            
            for route in routes:
                for i in range(len(route) - 1):
                    if (route[i] < pheromone_matrix.shape[0] and 
                        route[i+1] < pheromone_matrix.shape[1]):
                        pheromone_matrix[route[i], route[i+1]] += delta_pheromone
                        pheromone_matrix[route[i+1], route[i]] += delta_pheromone  # Symmetric
    
    def _apply_local_search(self, routes: List[List[int]], instance_idx: int) -> List[List[int]]:
        """Apply simple 2-opt local search to improve solution"""
        # Use the 2-opt method from NN2optSolver
        improved_routes = []
        
        for route in routes:
            if len(route) > 4:  # Only apply if route has enough nodes
                improved_route = self.nn_solver._time_aware_2opt(route, instance_idx)
                improved_routes.append(improved_route)
            else:
                improved_routes.append(route)
        
        return improved_routes
    
    def _create_distance_matrix(self, locations: np.ndarray) -> np.ndarray:
        """Create distance matrix from locations"""
        n = len(locations)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((locations[i] - locations[j])**2))
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        return dist_matrix