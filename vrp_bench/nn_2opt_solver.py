import time
import random
import numpy as np
from typing import List, Dict, Tuple
from vrp_base import VRPSolverBase
from travel_time_generator import sample_travel_time


class NN2optSolver(VRPSolverBase):
    """Highly optimized Nearest Neighbor + 2-opt solver with proper time constraint handling"""
    
    def __init__(self, data: Dict):
        """Initialize with problem data"""
        super().__init__(data)
        
        # Optimized configuration for speed
        self.max_2opt_iterations = 3  # Reduced for speed
        self.improvement_threshold = 0.01  # Lowered threshold for faster acceptance
        self.max_customers_per_route = 15  # Limit route length for 2-opt performance
        self.use_fast_2opt = True  # Use simplified 2-opt
        
        # Debug flags
        self.debug = False
    
    def solve_instance(self, instance_idx: int, num_realizations: int = 3) -> Dict:
        """Solve a single VRP instance with proper time constraint handling"""
        start_time = time.time()
        
        try:
            # Get basic instance data
            num_vehicles = self._get_num_vehicles(instance_idx)
            depots, customers = self.get_depots_and_customers(instance_idx)
            demands = self._get_demands(instance_idx)
            capacities = self._get_vehicle_capacities(instance_idx)
            
            # Get time constraints
            time_windows = self.get_time_windows(instance_idx)
            appear_times = self.get_appear_times(instance_idx)
            
            if self.debug:
                print(f"\n--- Instance {instance_idx} Debug Info ---")
                print(f"Num vehicles: {num_vehicles}")
                print(f"Depots: {depots}")
                print(f"Customers: {customers[:10]}... (total: {len(customers)})")
                print(f"Sample demands: {demands[:10]}... (total: {len(demands)})")
                print(f"Vehicle capacities: {capacities}")
                print(f"Time windows: {len(time_windows)} customers have time windows")
                print(f"Appear times: {len(appear_times)} customers have appear times")
            
            if len(customers) == 0:
                return self._create_empty_result()
            
            # Build routes using time-aware nearest neighbor
            routes = self._time_aware_nn_construction(instance_idx, num_vehicles, depots, customers)
            
            if self.debug:
                print(f"Constructed routes: {len(routes)}")
                for i, route in enumerate(routes[:3]):  # Show first 3 routes
                    route_demand = sum(demands[node] for node in route[1:-1] if node < len(demands))
                    print(f"Route {i}: length={len(route)}, demand={route_demand}")
            
            # Apply fast 2-opt improvement (with time constraint awareness)
            improved_routes = []
            for route in routes:
                if self.use_fast_2opt and len(route) > 4 and len(route) <= self.max_customers_per_route:
                    improved_route = self._time_aware_2opt(route, instance_idx)
                else:
                    improved_route = route
                improved_routes.append(improved_route)
            
            # Calculate solution cost with debugging
            result = self.calculate_solution_cost_debug(improved_routes, instance_idx, num_realizations)
            
            # Add runtime
            result['runtime'] = time.time() - start_time
            
            if self.debug:
                print(f"Final result: CVR={result['cvr']:.1f}%, Feasibility={result['feasibility']}")
                print(f"TW Violations: {result.get('time_window_violations', 0)}")
                print(f"Routes created: {len(improved_routes)}")
                print("----------------------------")
            
            return result
            
        except Exception as e:
            print(f"FastNN2opt Error: {e}")
            import traceback
            traceback.print_exc()
            return self._create_empty_result()
    
    def calculate_solution_cost_debug(self, routes: List[List[int]], instance_idx: int, 
                                   num_realizations: int = 3) -> Dict:
        """Calculate comprehensive solution cost with debugging"""
        # Calculate stochastic costs
        total_cost, waiting_time, robustness = self._calculate_stochastic_cost(
            routes, instance_idx, num_realizations
        )
        
        # Check feasibility with detailed time constraint checking
        cvr, feasibility, tw_violations = self._check_feasibility_debug(routes, instance_idx)
        
        return {
            'total_cost': total_cost,
            'waiting_time': waiting_time,
            'cvr': cvr,
            'feasibility': feasibility,
            'robustness': robustness,
            'routes': routes,
            'time_window_violations': tw_violations
        }
    
    def _check_feasibility_debug(self, routes: List[List[int]], instance_idx: int) -> Tuple[float, float, int]:
        """Debug version of feasibility check with proper time constraint handling"""
        # Get instance data
        demands = self._get_demands(instance_idx)
        capacities = self._get_vehicle_capacities(instance_idx)
        depots, customers = self.get_depots_and_customers(instance_idx)
        time_windows = self.get_time_windows(instance_idx)
        appear_times = self.get_appear_times(instance_idx)
        distance_dict = self.distance_dicts[instance_idx] if instance_idx < len(self.distance_dicts) else {}
        
        if self.debug:
            print(f"\n--- Feasibility Check Debug ---")
            print(f"Customers to serve: {len(customers)}")
            print(f"Number of routes: {len(routes)}")
            print(f"Time windows defined: {len(time_windows)}")
            print(f"Appear times defined: {len(appear_times)}")
        
        total_violations = 0
        customers_served = 0
        capacity_violations = 0
        time_window_violations = 0
        appear_time_violations = 0
        customer_visits = {}
        
        # Process each route
        for route_idx, route in enumerate(routes):
            if len(route) <= 2:  # Skip empty routes
                continue
            
            # Get vehicle capacity
            vehicle_capacity = capacities[route_idx] if route_idx < len(capacities) else (capacities[0] if len(capacities) > 0 else 100)
            route_demand = 0
            route_customers = 0
            current_time = 0
            
            # Track route timing for debugging
            route_timing = [f"Route {route_idx} Timing:"]
            
            for i, node in enumerate(route):
                if node in customers:
                    customers_served += 1
                    route_customers += 1
                    customer_visits[node] = customer_visits.get(node, 0) + 1
                    
                    if node < len(demands):
                        route_demand += demands[node]
                
                # Calculate timing
                if i > 0:
                    prev_node = route[i-1]
                    travel_time = sample_travel_time(prev_node, node, distance_dict, current_time)
                    current_time += travel_time
                    route_timing.append(f"  {prev_node}->{node}: arrive at {current_time:.1f}")
                
                # Check appear time constraint
                if node in appear_times and node not in depots:
                    if current_time < appear_times[node]:
                        appear_time_violations += 1
                        route_timing.append(f"  Node {node}: APPEAR TIME VIOLATION (arrived {current_time:.1f}, appears at {appear_times[node]:.1f})")
                
                # Check time window constraint
                if node in time_windows and node not in depots:
                    start_time, end_time = time_windows[node]
                    
                    if current_time > end_time:
                        time_window_violations += 1
                        route_timing.append(f"  Node {node}: TIME WINDOW VIOLATION (arrived {current_time:.1f}, window closes at {end_time:.1f})")
                    elif current_time < start_time:
                        route_timing.append(f"  Node {node}: WAIT until {start_time:.1f} (early arrival)")
                        current_time = start_time
            
            # Check capacity violation
            if route_demand > vehicle_capacity * 1.001:  # Small tolerance
                capacity_violations += 1
                route_timing.append(f"  CAPACITY VIOLATION: demand={route_demand:.1f}, capacity={vehicle_capacity:.1f}")
            
            if self.debug and (capacity_violations > 0 or time_window_violations > 0 or appear_time_violations > 0):
                for line in route_timing:
                    print(line)
        
        # Count additional violations
        total_violations = capacity_violations + time_window_violations + appear_time_violations
        
        # Multiple visits
        for customer, count in customer_visits.items():
            if count > 1:
                total_violations += (count - 1)
                if self.debug:
                    print(f"Customer {customer} visited {count} times (violation!)")
        
        # Unserved customers
        unserved = len(customers) - customers_served
        total_violations += unserved
        
        if self.debug:
            print(f"Total customers: {len(customers)}")
            print(f"Customers served: {customers_served}")
            print(f"Unserved customers: {unserved}")
            print(f"Capacity violations: {capacity_violations}")
            print(f"Time window violations: {time_window_violations}")
            print(f"Appear time violations: {appear_time_violations}")
            print(f"Total violations: {total_violations}")
        
        # Calculate metrics
        if len(customers) > 0:
            cvr = (total_violations / len(customers)) * 100
        else:
            cvr = 0.0 if total_violations == 0 else 100.0
        
        feasibility = max(0.0, 1.0 - (total_violations / len(customers)))
        
        if self.debug:
            print(f"CVR: {cvr}%")
            print(f"Feasibility: {feasibility}")
            print("--------------------------------")
        
        return cvr, feasibility, time_window_violations
    
    def _time_aware_nn_construction(self, instance_idx: int, num_vehicles: int, 
                                  depots: np.ndarray, customers: np.ndarray) -> List[List[int]]:
        """Time-aware nearest neighbor construction that considers appear times and time windows"""
        routes = []
        unvisited = set(customers)
        
        if instance_idx >= len(self.dist_matrices):
            return [[depot, depot] for depot in depots[:num_vehicles]]
        
        dist_matrix = self.dist_matrices[instance_idx]
        demands = self._get_demands(instance_idx)
        capacities = self._get_vehicle_capacities(instance_idx)
        time_windows = self.get_time_windows(instance_idx)
        appear_times = self.get_appear_times(instance_idx)
        distance_dict = self.distance_dicts[instance_idx] if instance_idx < len(self.distance_dicts) else {}
        
        # Ensure we have enough capacity information
        if len(capacities) == 0:
            capacities = np.array([100] * num_vehicles)
        elif len(capacities) < num_vehicles:
            capacities = np.concatenate([capacities, [capacities[0]] * (num_vehicles - len(capacities))])
        
        # Create routes for each vehicle
        for vehicle in range(num_vehicles):
            if not unvisited:
                break
            
            # Get vehicle capacity
            vehicle_capacity = capacities[vehicle]
            
            # Start from the appropriate depot
            if vehicle < len(depots):
                start_depot = depots[vehicle]
            else:
                start_depot = depots[0] if len(depots) > 0 else 0
            
            route = [start_depot]
            current = start_depot
            route_demand = 0
            current_time = 0
            
            # Add customers using time-aware nearest neighbor
            while unvisited:
                # Find feasible customers (capacity, appear time, time window)
                feasible_customers = []
                
                for customer in unvisited:
                    if customer < len(demands):
                        customer_demand = demands[customer]
                        
                        # Check capacity constraint
                        if route_demand + customer_demand > vehicle_capacity:
                            continue
                        
                        # Estimate arrival time at customer
                        travel_time = sample_travel_time(current, customer, distance_dict, current_time)
                        arrival_time = current_time + travel_time
                        
                        # Check appear time constraint
                        if customer in appear_times:
                            if arrival_time < appear_times[customer]:
                                continue  # Cannot visit before customer appears
                        
                        # Check time window constraint
                        if customer in time_windows:
                            start_time, end_time = time_windows[customer]
                            if arrival_time > end_time:
                                continue  # Would arrive too late
                        
                        feasible_customers.append(customer)
                
                if not feasible_customers:
                    break
                
                # Find nearest among feasible customers
                nearest = None
                min_dist = float('inf')
                
                for customer in feasible_customers:
                    if (current < len(dist_matrix) and customer < len(dist_matrix) and 
                        customer < len(dist_matrix[current])):
                        dist = dist_matrix[current, customer]
                        if dist < min_dist:
                            min_dist = dist
                            nearest = customer
                
                if nearest is None:
                    break
                
                # Add customer to route and update time
                route.append(nearest)
                unvisited.remove(nearest)
                route_demand += demands[nearest] if nearest < len(demands) else 0
                
                # Update current time
                travel_time = sample_travel_time(current, nearest, distance_dict, current_time)
                current_time += travel_time
                
                # Check if we need to wait for customer to appear
                if nearest in appear_times:
                    if current_time < appear_times[nearest]:
                        current_time = appear_times[nearest]
                
                # Check if we need to wait for time window
                if nearest in time_windows:
                    start_time, end_time = time_windows[nearest]
                    if current_time < start_time:
                        current_time = start_time
                
                current = nearest
            
            # Return to depot
            route.append(start_depot)
            routes.append(route)
        
        # Create single-customer routes for remaining customers
        # But only if they can be served respecting all constraints
        while unvisited:
            start_depot = depots[0] if len(depots) > 0 else 0
            customer = unvisited.pop()
            
            # Check if this customer can be served at all
            can_serve = True
            
            # Check appear time and time window
            if customer in appear_times and customer in time_windows:
                if appear_times[customer] > time_windows[customer][1]:
                    can_serve = False  # Customer appears after their time window closes
            
            if can_serve:
                routes.append([start_depot, customer, start_depot])
            # If can't serve, leave unserved (will be counted as violation)
        
        return routes
    
    def _time_aware_2opt(self, route: List[int], instance_idx: int) -> List[int]:
        """Time-aware 2-opt improvement that respects time constraints"""
        if len(route) <= 4:  # Need at least 5 nodes for 2-opt
            return route
        
        if instance_idx >= len(self.dist_matrices):
            return route
        
        best_route = route.copy()
        dist_matrix = self.dist_matrices[instance_idx]
        distance_dict = self.distance_dicts[instance_idx] if instance_idx < len(self.distance_dicts) else {}
        
        # Get time constraints
        time_windows = self.get_time_windows(instance_idx)
        appear_times = self.get_appear_times(instance_idx)
        
        # Calculate initial cost
        best_cost = self._calculate_route_cost_with_constraints(best_route, distance_dict, time_windows, appear_times)
        
        if best_cost == float('inf'):  # Route is infeasible
            return best_route
        
        # Very limited iterations for speed
        max_iterations = min(self.max_2opt_iterations, len(route) // 2)
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try limited 2-opt swaps
            for i in range(1, min(len(best_route) - 2, 5)):  # Limit starting points
                for j in range(i + 2, min(len(best_route), i + 6)):  # Limit j range
                    # Create new route with 2-opt swap
                    new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                    
                    # Calculate cost with constraints
                    new_cost = self._calculate_route_cost_with_constraints(new_route, distance_dict, time_windows, appear_times)
                    
                    # Accept any improvement (lowered threshold)
                    if new_cost < best_cost:
                        best_route = new_route
                        best_cost = new_cost
                        improved = True
                        break  # Accept first improvement
                
                if improved:
                    break  # Exit outer loop on improvement
            
            # Stop if no improvement
            if not improved:
                break
        
        return best_route
    
    def _calculate_route_cost_with_constraints(self, route: List[int], distance_dict: Dict, 
                                             time_windows: Dict, appear_times: Dict) -> float:
        """Calculate route cost with time constraint checking"""
        if len(route) <= 1:
            return 0
        
        total_cost = 0
        current_time = 0
        
        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]
            
            # Get travel time
            travel_time = sample_travel_time(current_node, next_node, distance_dict, current_time)
            current_time += travel_time
            total_cost += travel_time
            
            # Check appear time constraint
            if next_node in appear_times:
                if current_time < appear_times[next_node]:
                    # Must wait for customer to appear
                    wait_time = appear_times[next_node] - current_time
                    current_time = appear_times[next_node]
                    total_cost += wait_time * 0.1  # Small penalty for waiting
            
            # Check time window constraint
            if next_node in time_windows:
                start_time, end_time = time_windows[next_node]
                
                if current_time > end_time:
                    return float('inf')  # Infeasible route
                elif current_time < start_time:
                    # Wait for time window to open
                    wait_time = start_time - current_time
                    current_time = start_time
                    total_cost += wait_time * 0.1  # Small penalty for waiting
        
        return total_cost