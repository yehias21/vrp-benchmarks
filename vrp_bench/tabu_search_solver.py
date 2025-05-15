import time
import random
import numpy as np
from typing import List, Dict, Tuple, Set
from vrp_base import VRPSolverBase
from travel_time_generator import sample_travel_time
from nn_2opt_solver import NN2optSolver


class TabuSearchSolver(VRPSolverBase):
    """Improved Tabu Search solver with better feasibility handling"""
    
    def __init__(self, data: Dict):
        """Initialize with problem data"""
        super().__init__(data)
        
        # Dynamic parameters based on problem size
        self.tabu_tenure_base = 3
        self.max_iterations_base = 20
        self.max_non_improving_base = 6
        
        # Constraint weights
        self.capacity_penalty = 10000
        self.time_window_penalty = 5000
        self.appear_time_penalty = 2000
        
        # Create initial solver for baseline
        self.nn_solver = NN2optSolver(data)
        self.nn_solver.debug = False
        
        # Debug flags
        self.debug = False
    
    def solve_instance(self, instance_idx: int, num_realizations: int = 3) -> Dict:
        """Solve using improved Tabu Search with feasibility focus"""
        start_time = time.time()
        
        try:
            # Get problem size to adjust parameters
            _, customers = self.get_depots_and_customers(instance_idx)
            problem_size = len(customers)
            
            # Scale parameters based on problem size
            self._adjust_parameters(problem_size)
            
            # Get initial solution from NN+2opt
            initial_result = self.nn_solver.solve_instance(instance_idx, 1)
            current_routes = initial_result['routes']
            
            if self.debug:
                print(f"\n--- Improved Tabu Search Instance {instance_idx} ---")
                print(f"Problem size: {problem_size} customers")
                print(f"Initial cost: {initial_result['total_cost']:.1f}")
                print(f"Initial CVR: {initial_result['cvr']:.1f}%")
            
            # Repair initial solution if needed
            current_routes = self._repair_solution(current_routes, instance_idx)
            
            # Apply enhanced Tabu Search
            best_routes = self._enhanced_tabu_search(current_routes, instance_idx)
            
            # Final repair pass
            best_routes = self._repair_solution(best_routes, instance_idx)
            
            # Calculate final solution cost
            result = self.calculate_solution_cost(best_routes, instance_idx, num_realizations)
            result['runtime'] = time.time() - start_time
            
            if self.debug:
                print(f"Final cost: {result['total_cost']:.1f}")
                print(f"Final CVR: {result['cvr']:.1f}%")
                print(f"Final feasibility: {result['feasibility']:.3f}")
                print("------------------------")
            
            return result
            
        except Exception as e:
            print(f"TabuSearch Error: {e}")
            import traceback
            traceback.print_exc()
            return self._create_empty_result()
    
    def _adjust_parameters(self, problem_size: int):
        """Dynamically adjust parameters based on problem size"""
        if problem_size <= 20:
            self.tabu_tenure = self.tabu_tenure_base
            self.max_iterations = self.max_iterations_base
            self.max_non_improving = self.max_non_improving_base
            self.max_moves_per_iteration = 30
        elif problem_size <= 50:
            self.tabu_tenure = self.tabu_tenure_base + 2
            self.max_iterations = self.max_iterations_base - 5
            self.max_non_improving = self.max_non_improving_base - 1
            self.max_moves_per_iteration = 25
        elif problem_size <= 100:
            self.tabu_tenure = self.tabu_tenure_base + 3
            self.max_iterations = self.max_iterations_base - 8
            self.max_non_improving = self.max_non_improving_base - 2
            self.max_moves_per_iteration = 20
        elif problem_size <= 200:
            self.tabu_tenure = self.tabu_tenure_base + 4
            self.max_iterations = self.max_iterations_base - 10
            self.max_non_improving = self.max_non_improving_base - 3
            self.max_moves_per_iteration = 15
        else:  # Large instances
            self.tabu_tenure = self.tabu_tenure_base + 5
            self.max_iterations = self.max_iterations_base - 12
            self.max_non_improving = self.max_non_improving_base - 4
            self.max_moves_per_iteration = 10
    
    def _repair_solution(self, routes: List[List[int]], instance_idx: int) -> List[List[int]]:
        """Repair solution to make it feasible"""
        repaired_routes = [route.copy() for route in routes]
        
        # Get data
        demands = self._get_demands(instance_idx)
        capacities = self._get_vehicle_capacities(instance_idx)
        time_windows = self.get_time_windows(instance_idx)
        appear_times = self.get_appear_times(instance_idx)
        depots, customers = self.get_depots_and_customers(instance_idx)
        distance_dict = self.distance_dicts[instance_idx] if instance_idx < len(self.distance_dicts) else {}
        
        # Repair capacity violations
        for route_idx, route in enumerate(repaired_routes):
            if len(route) <= 2:
                continue
            
            # Check capacity
            route_demand = sum(demands[node] for node in route[1:-1] if node < len(demands))
            vehicle_capacity = capacities[route_idx] if route_idx < len(capacities) else (capacities[0] if len(capacities) > 0 else 100)
            
            # If over capacity, remove furthest customers
            while route_demand > vehicle_capacity and len(route) > 3:
                # Find customer with highest removal cost
                worst_customer = None
                worst_savings = -float('inf')
                
                for i in range(1, len(route) - 1):
                    customer = route[i]
                    if customer not in customers:
                        continue
                    
                    # Calculate removal savings
                    if i > 0 and i < len(route) - 1:
                        prev_node = route[i-1]
                        next_node = route[i+1]
                        
                        # Current cost
                        current_cost = (sample_travel_time(prev_node, customer, distance_dict, 0) +
                                      sample_travel_time(customer, next_node, distance_dict, 0))
                        # Direct cost
                        direct_cost = sample_travel_time(prev_node, next_node, distance_dict, 0)
                        
                        savings = current_cost - direct_cost
                        
                        if savings > worst_savings:
                            worst_savings = savings
                            worst_customer = i
                
                if worst_customer is not None:
                    removed_node = route.pop(worst_customer)
                    route_demand -= demands[removed_node] if removed_node < len(demands) else 0
                else:
                    break
            
            repaired_routes[route_idx] = route
        
        # Collect unserved customers
        served_customers = set()
        for route in repaired_routes:
            for node in route:
                if node in customers:
                    served_customers.add(node)
        
        unserved = [c for c in customers if c not in served_customers]
        
        # Try to insert unserved customers
        for customer in unserved:
            best_route = None
            best_position = None
            best_cost_increase = float('inf')
            
            for route_idx, route in enumerate(repaired_routes):
                # Check capacity
                route_demand = sum(demands[node] for node in route[1:-1] if node < len(demands))
                customer_demand = demands[customer] if customer < len(demands) else 0
                vehicle_capacity = capacities[route_idx] if route_idx < len(capacities) else (capacities[0] if len(capacities) > 0 else 100)
                
                if route_demand + customer_demand > vehicle_capacity:
                    continue
                
                # Try different insertion positions
                for pos in range(1, len(route)):
                    # Check time feasibility
                    if self._check_time_feasibility(route, customer, pos, instance_idx, time_windows, appear_times, distance_dict):
                        # Calculate insertion cost
                        if pos > 0 and pos < len(route):
                            prev_cost = sample_travel_time(route[pos-1], route[pos], distance_dict, 0)
                            new_cost1 = sample_travel_time(route[pos-1], customer, distance_dict, 0)
                            new_cost2 = sample_travel_time(customer, route[pos], distance_dict, 0)
                            cost_increase = new_cost1 + new_cost2 - prev_cost
                            
                            if cost_increase < best_cost_increase:
                                best_cost_increase = cost_increase
                                best_route = route_idx
                                best_position = pos
            
            # Insert customer at best position
            if best_route is not None and best_position is not None:
                repaired_routes[best_route].insert(best_position, customer)
        
        return repaired_routes
    
    def _check_time_feasibility(self, route: List[int], customer: int, position: int, 
                               instance_idx: int, time_windows: Dict, appear_times: Dict, 
                               distance_dict: Dict) -> bool:
        """Check if inserting customer at position maintains time feasibility"""
        # Create temporary route
        temp_route = route[:position] + [customer] + route[position:]
        
        current_time = 0
        for i in range(len(temp_route) - 1):
            current_node = temp_route[i]
            next_node = temp_route[i + 1]
            
            # Travel time
            travel_time = sample_travel_time(current_node, next_node, distance_dict, current_time)
            current_time += travel_time
            
            # Check appear time
            if next_node in appear_times:
                if current_time < appear_times[next_node]:
                    current_time = appear_times[next_node]
            
            # Check time window
            if next_node in time_windows:
                start_time, end_time = time_windows[next_node]
                if current_time > end_time:
                    return False
                elif current_time < start_time:
                    current_time = start_time
        
        return True
    
    def _enhanced_tabu_search(self, initial_routes: List[List[int]], instance_idx: int) -> List[List[int]]:
        """Enhanced Tabu Search with feasibility focus"""
        current_routes = [route.copy() for route in initial_routes]
        best_routes = [route.copy() for route in initial_routes]
        
        # Calculate initial costs with strong feasibility focus
        current_cost = self._calculate_enhanced_objective(current_routes, instance_idx)
        best_cost = current_cost
        
        # Initialize tabu list
        tabu_list: Set[Tuple] = set()
        
        # Track improvements
        iterations_without_improvement = 0
        feasibility_improvement_counter = 0
        
        for iteration in range(self.max_iterations):
            if self.debug:
                print(f"Iteration {iteration}: best_cost={best_cost:.1f}")
            
            # Find best move
            best_move = None
            best_move_cost = float('inf')
            best_move_routes = None
            best_move_is_improving = False
            
            # Evaluate moves
            moves_evaluated = 0
            
            for move_type in ['swap', 'relocate', 'exchange']:
                if moves_evaluated >= self.max_moves_per_iteration:
                    break
                
                moves = self._generate_feasible_moves(current_routes, move_type, instance_idx)
                
                for move in moves:
                    if moves_evaluated >= self.max_moves_per_iteration:
                        break
                    
                    # Check if move is tabu (unless it's aspiration criterion)
                    if move in tabu_list:
                        continue
                    
                    # Apply move and evaluate
                    new_routes = self._apply_move(current_routes, move)
                    new_cost = self._calculate_enhanced_objective(new_routes, instance_idx)
                    
                    # Aspiration criterion: accept if better than best
                    is_improving = new_cost < best_cost * 1.01  # Small tolerance
                    
                    # Update best move
                    if is_improving or new_cost < best_move_cost:
                        best_move = move
                        best_move_cost = new_cost
                        best_move_routes = new_routes
                        best_move_is_improving = is_improving
                    
                    moves_evaluated += 1
            
            # No valid move found
            if best_move is None:
                break
            
            # Apply best move
            current_routes = best_move_routes
            current_cost = best_move_cost
            
            # Update tabu list
            tabu_list.add(best_move)
            if len(tabu_list) > self.tabu_tenure:
                tabu_list = set(list(tabu_list)[-self.tabu_tenure:])
            
            # Update global best
            if best_move_is_improving:
                best_cost = current_cost
                best_routes = [route.copy() for route in current_routes]
                iterations_without_improvement = 0
                feasibility_improvement_counter += 1
            else:
                iterations_without_improvement += 1
            
            # Early stopping
            if iterations_without_improvement >= self.max_non_improving:
                break
            
            # Diversification (reset to best solution after long stagnation)
            if iterations_without_improvement >= self.max_non_improving - 2:
                current_routes = [route.copy() for route in best_routes]
                current_cost = best_cost
        
        return best_routes
    
    def _generate_feasible_moves(self, routes: List[List[int]], move_type: str, instance_idx: int) -> List[Tuple]:
        """Generate moves with feasibility checks"""
        moves = []
        
        if move_type == 'swap':
            # Customer swap within routes (feasibility-aware)
            for route_idx, route in enumerate(routes):
                customers_in_route = [node for node in route if node in self.customer_indices[instance_idx]]
                
                for i in range(min(len(customers_in_route), 4)):
                    for j in range(i + 1, min(len(customers_in_route), i + 4)):
                        customer1 = customers_in_route[i]
                        customer2 = customers_in_route[j]
                        pos1 = route.index(customer1)
                        pos2 = route.index(customer2)
                        
                        # Quick feasibility check
                        if self._quick_swap_feasibility(route, pos1, pos2, instance_idx):
                            moves.append(('swap', route_idx, pos1, pos2))
        
        elif move_type == 'relocate':
            # Relocate with feasibility checks
            for from_route in range(len(routes)):
                customers_in_route = [node for node in routes[from_route] 
                                    if node in self.customer_indices[instance_idx]]
                
                for customer in customers_in_route[:3]:
                    pos = routes[from_route].index(customer)
                    
                    for to_route in range(len(routes)):
                        if from_route == to_route:
                            continue
                        
                        # Try a few insertion positions with feasibility checks
                        for insert_pos in range(1, min(len(routes[to_route]), 4)):
                            if self._quick_relocate_feasibility(routes, from_route, pos, to_route, insert_pos, instance_idx):
                                moves.append(('relocate', from_route, pos, to_route, insert_pos))
        
        elif move_type == 'exchange':
            # Exchange customers between routes
            for route1 in range(len(routes)):
                for route2 in range(route1 + 1, len(routes)):
                    customers1 = [n for n in routes[route1] if n in self.customer_indices[instance_idx]]
                    customers2 = [n for n in routes[route2] if n in self.customer_indices[instance_idx]]
                    
                    if customers1 and customers2:
                        # Simple 1-1 exchange
                        pos1 = routes[route1].index(customers1[0])
                        pos2 = routes[route2].index(customers2[0])
                        
                        if self._quick_exchange_feasibility(routes, route1, pos1, route2, pos2, instance_idx):
                            moves.append(('exchange', route1, pos1, route2, pos2))
        
        return moves[:self.max_moves_per_iteration]
    
    def _quick_swap_feasibility(self, route: List[int], pos1: int, pos2: int, instance_idx: int) -> bool:
        """Quick feasibility check for swap move"""
        # Create temporary route
        temp_route = route.copy()
        temp_route[pos1], temp_route[pos2] = temp_route[pos2], temp_route[pos1]
        
        # Quick time window check around affected positions
        time_windows = self.get_time_windows(instance_idx)
        distance_dict = self.distance_dicts[instance_idx] if instance_idx < len(self.distance_dicts) else {}
        
        # Check time feasibility for affected segments
        for check_pos in [max(0, min(pos1, pos2) - 1), min(pos1, pos2), max(pos1, pos2), min(len(temp_route) - 1, max(pos1, pos2) + 1)]:
            if check_pos >= len(temp_route) - 1:
                continue
            
            node = temp_route[check_pos]
            next_node = temp_route[check_pos + 1]
            
            if node in time_windows or next_node in time_windows:
                # Detailed check only for time window nodes
                return self._detailed_time_check(temp_route, check_pos, instance_idx)
        
        return True
    
    def _quick_relocate_feasibility(self, routes: List[List[int]], from_route: int, from_pos: int, 
                                   to_route: int, to_pos: int, instance_idx: int) -> bool:
        """Quick feasibility check for relocate move"""
        demands = self._get_demands(instance_idx)
        capacities = self._get_vehicle_capacities(instance_idx)
        
        customer = routes[from_route][from_pos]
        customer_demand = demands[customer] if customer < len(demands) else 0
        
        # Quick capacity check
        to_route_demand = sum(demands[node] for node in routes[to_route][1:-1] if node < len(demands))
        to_vehicle_capacity = capacities[to_route] if to_route < len(capacities) else (capacities[0] if len(capacities) > 0 else 100)
        
        if to_route_demand + customer_demand > to_vehicle_capacity:
            return False
        
        # Quick time window check
        time_windows = self.get_time_windows(instance_idx)
        if customer in time_windows:
            # Do more detailed time check only if customer has time window
            return self._check_time_feasibility(routes[to_route], customer, to_pos, instance_idx, 
                                              time_windows, self.get_appear_times(instance_idx), 
                                              self.distance_dicts[instance_idx])
        
        return True
    
    def _quick_exchange_feasibility(self, routes: List[List[int]], route1: int, pos1: int, 
                                   route2: int, pos2: int, instance_idx: int) -> bool:
        """Quick feasibility check for exchange move"""
        demands = self._get_demands(instance_idx)
        capacities = self._get_vehicle_capacities(instance_idx)
        
        customer1 = routes[route1][pos1]
        customer2 = routes[route2][pos2]
        
        demand1 = demands[customer1] if customer1 < len(demands) else 0
        demand2 = demands[customer2] if customer2 < len(demands) else 0
        
        # Check capacity constraints
        route1_demand = sum(demands[node] for node in routes[route1][1:-1] if node < len(demands))
        route2_demand = sum(demands[node] for node in routes[route2][1:-1] if node < len(demands))
        
        cap1 = capacities[route1] if route1 < len(capacities) else (capacities[0] if len(capacities) > 0 else 100)
        cap2 = capacities[route2] if route2 < len(capacities) else (capacities[0] if len(capacities) > 0 else 100)
        
        new_demand1 = route1_demand - demand1 + demand2
        new_demand2 = route2_demand - demand2 + demand1
        
        return new_demand1 <= cap1 and new_demand2 <= cap2
    
    def _detailed_time_check(self, route: List[int], start_pos: int, instance_idx: int) -> bool:
        """Detailed time feasibility check from start_pos onwards"""
        time_windows = self.get_time_windows(instance_idx)
        appear_times = self.get_appear_times(instance_idx)
        distance_dict = self.distance_dicts[instance_idx] if instance_idx < len(self.distance_dicts) else {}
        
        current_time = 0
        
        # Calculate time up to start_pos
        for i in range(min(start_pos + 1, len(route) - 1)):
            current_node = route[i]
            next_node = route[i + 1]
            
            travel_time = sample_travel_time(current_node, next_node, distance_dict, current_time)
            current_time += travel_time
            
            if next_node in appear_times:
                current_time = max(current_time, appear_times[next_node])
            
            if next_node in time_windows:
                start_time, end_time = time_windows[next_node]
                if current_time > end_time:
                    return False
                elif current_time < start_time:
                    current_time = start_time
        
        return True
    
    def _apply_move(self, routes: List[List[int]], move: Tuple) -> List[List[int]]:
        """Apply a move to create new solution"""
        new_routes = [route.copy() for route in routes]
        
        if move[0] == 'swap':
            _, route_idx, pos1, pos2 = move
            new_routes[route_idx][pos1], new_routes[route_idx][pos2] = \
                new_routes[route_idx][pos2], new_routes[route_idx][pos1]
        
        elif move[0] == 'relocate':
            _, from_route, from_pos, to_route, to_pos = move
            customer = new_routes[from_route].pop(from_pos)
            new_routes[to_route].insert(to_pos, customer)
        
        elif move[0] == 'exchange':
            _, route1, pos1, route2, pos2 = move
            customer1 = new_routes[route1][pos1]
            customer2 = new_routes[route2][pos2]
            new_routes[route1][pos1] = customer2
            new_routes[route2][pos2] = customer1
        
        return new_routes
    
    def _calculate_enhanced_objective(self, routes: List[List[int]], instance_idx: int) -> float:
        """Calculate objective with strong feasibility focus"""
        # Base cost
        total_cost = 0
        
        # Get data
        demands = self._get_demands(instance_idx)
        capacities = self._get_vehicle_capacities(instance_idx)
        time_windows = self.get_time_windows(instance_idx)
        appear_times = self.get_appear_times(instance_idx)
        distance_dict = self.distance_dicts[instance_idx] if instance_idx < len(self.distance_dicts) else {}
        depots, customers = self.get_depots_and_customers(instance_idx)
        
        # Calculate base costs and violations
        capacity_violations = 0
        time_window_violations = 0
        appear_time_violations = 0
        
        for route_idx, route in enumerate(routes):
            if len(route) <= 2:
                continue
            
            # Route cost and time tracking
            route_cost = 0
            current_time = 0
            route_demand = 0
            
            for i in range(len(route) - 1):
                current_node = route[i]
                next_node = route[i + 1]
                
                # Add demand
                if next_node in customers and next_node < len(demands):
                    route_demand += demands[next_node]
                
                # Travel time and cost
                travel_time = sample_travel_time(current_node, next_node, distance_dict, current_time)
                current_time += travel_time
                route_cost += travel_time
                
                # Check appear time violation
                if next_node in appear_times:
                    if current_time < appear_times[next_node]:
                        appear_time_violations += 1
                        current_time = appear_times[next_node]
                
                # Check time window violation
                if next_node in time_windows:
                    start_time, end_time = time_windows[next_node]
                    if current_time > end_time:
                        time_window_violations += 1
                    elif current_time < start_time:
                        current_time = start_time
            
            # Check capacity violation
            vehicle_capacity = capacities[route_idx] if route_idx < len(capacities) else (capacities[0] if len(capacities) > 0 else 100)
            if route_demand > vehicle_capacity:
                capacity_violations += 1
            
            total_cost += route_cost
        
        # Calculate unserved customers
        served_customers = set()
        for route in routes:
            for node in route:
                if node in customers:
                    served_customers.add(node)
        
        unserved_customers = len([c for c in customers if c not in served_customers])
        
        # Apply heavy penalties for violations
        penalty = 0
        penalty += capacity_violations * self.capacity_penalty
        penalty += time_window_violations * self.time_window_penalty
        penalty += appear_time_violations * self.appear_time_penalty
        penalty += unserved_customers * self.capacity_penalty * 2  # Very high penalty for unserved
        
        return total_cost + penalty