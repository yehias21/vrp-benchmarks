import time
import random
import numpy as np
from typing import List, Dict, Tuple
from vrp_base import VRPSolverBase
from travel_time_generator import sample_travel_time
from nn_2opt_solver import NN2optSolver

# Try to import OR-Tools with better error handling
OR_TOOLS_AVAILABLE = False
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    OR_TOOLS_AVAILABLE = True
except Exception as e:
    print(f"OR-Tools not available: {e}")
    pass


class ORToolsSolver(VRPSolverBase):
    """Improved OR-Tools solver for version 9.8.3296"""
    
    def __init__(self, data: Dict):
        """Initialize with problem data"""
        super().__init__(data)
        
        # Create fallback solver
        self.fallback_solver = NN2optSolver(data)
        self.fallback_solver.debug = False
        
        # Optimized solver parameters
        self.time_limit = 1  # Increased to 5 seconds for better solutions
        
        # Debug flag
        self.debug = False
    
    def solve_instance(self, instance_idx: int, num_realizations: int = 3) -> Dict:
        """Solve using OR-Tools"""
        if not OR_TOOLS_AVAILABLE:
            return self._solve_with_fallback(instance_idx, num_realizations)
        
        start_time = time.time()
        
        try:
            result = self._solve_with_or_tools(instance_idx, num_realizations)
            result['runtime'] = time.time() - start_time
            
            if self.debug:
                print(f"OR-Tools Instance {instance_idx}: Cost={result['total_cost']:.1f}, CVR={result['cvr']:.1f}%")
            
            return result
            
        except Exception as e:
            print(f"OR-Tools Error: {e}")
            import traceback
            traceback.print_exc()
            return self._solve_with_fallback(instance_idx, num_realizations)
    
    def _solve_with_or_tools(self, instance_idx: int, num_realizations: int) -> Dict:
        """Solve using real OR-Tools"""
        try:
            # Get problem data
            num_vehicles = self._get_num_vehicles(instance_idx)
            depots, customers = self.get_depots_and_customers(instance_idx)
            demands = self._get_demands(instance_idx)
            capacities = self._get_vehicle_capacities(instance_idx)
            
            if len(customers) == 0:
                return self._create_empty_result()
            
            # Create data model
            data = self._create_data_model(instance_idx, num_vehicles, depots, customers, demands, capacities)
            
            # Create the routing index manager
            # Fix for the API error: Use proper argument types
            manager = pywrapcp.RoutingIndexManager(
                data['num_locations'],
                data['num_vehicles'],
                data['depot']  # Single depot index
            )
            
            # Create Routing Model
            routing = pywrapcp.RoutingModel(manager)
            
            # Create distance callback
            def distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return data['distance_matrix'][from_node][to_node]
            
            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            
            # Add Capacity constraint
            def demand_callback(from_index):
                from_node = manager.IndexToNode(from_index)
                return data['demands'][from_node]
            
            demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
            routing.AddDimensionWithVehicleCapacity(
                demand_callback_index,
                0,  # null capacity slack
                data['vehicle_capacities'],  # vehicle maximum capacities
                True,  # start cumul to zero
                'Capacity'
            )
            
            # Setting first solution heuristic
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.SAVINGS
            search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            search_parameters.time_limit.seconds = self.time_limit
            search_parameters.log_search = False
            
            # Solve the problem
            solution = routing.SolveWithParameters(search_parameters)
            
            if solution:
                # Extract solution
                routes = self._extract_routes(solution, routing, manager, data['num_vehicles'])
                
                # Convert routes back to original indices
                original_routes = self._convert_to_original_indices(routes, depots, customers)
                
                # Calculate final solution cost
                return self.calculate_solution_cost(original_routes, instance_idx, num_realizations)
            else:
                # No solution found, use fallback
                return self._solve_with_fallback(instance_idx, num_realizations)
        
        except Exception as e:
            print(f"OR-Tools solving error: {e}")
            import traceback
            traceback.print_exc()
            return self._solve_with_fallback(instance_idx, num_realizations)
    
    def _create_data_model(self, instance_idx: int, num_vehicles: int, depots: np.ndarray, 
                          customers: np.ndarray, demands: np.ndarray, capacities: np.ndarray) -> Dict:
        """Create data model for OR-Tools"""
        # Get instance locations
        if instance_idx < len(self.locations):
            if isinstance(self.locations, list):
                locations = self.locations[instance_idx]
            else:
                if len(self.locations.shape) == 3:
                    locations = self.locations[instance_idx]
                else:
                    locations = self.locations
        else:
            # Fallback
            return self._get_fallback_data()
        
        # Create node list: depot first, then customers
        num_locations = 1 + len(customers)  # Simplified: single depot
        node_locations = []
        
        # Add depot (use first depot)
        depot_idx = depots[0] if len(depots) > 0 else 0
        node_locations.append(locations[depot_idx])
        
        # Add customers
        for customer in customers:
            if customer < len(locations):
                node_locations.append(locations[customer])
        
        # Create distance matrix
        distance_matrix = []
        for i in range(num_locations):
            row = []
            for j in range(num_locations):
                if i == j:
                    row.append(0)
                else:
                    if i < len(node_locations) and j < len(node_locations):
                        dist = np.sqrt(np.sum((node_locations[i] - node_locations[j])**2))
                        row.append(int(dist * 100))  # Scale and convert to int
                    else:
                        row.append(999999)
            distance_matrix.append(row)
        
        # Create demands array
        demands_array = [0]  # Depot has 0 demand
        for customer in customers:
            if customer < len(demands):
                demands_array.append(max(1, int(demands[customer])))
            else:
                demands_array.append(1)
        
        # Create vehicle capacities array
        vehicle_capacities = []
        for i in range(num_vehicles):
            if i < len(capacities):
                vehicle_capacities.append(max(10, int(capacities[i])))
            else:
                vehicle_capacities.append(max(10, int(capacities[0]) if len(capacities) > 0 else 100))
        
        return {
            'distance_matrix': distance_matrix,
            'demands': demands_array,
            'vehicle_capacities': vehicle_capacities,
            'num_vehicles': num_vehicles,
            'num_locations': num_locations,
            'depot': 0  # Depot is at index 0
        }
    
    def _extract_routes(self, solution, routing, manager, num_vehicles: int) -> List[List[int]]:
        """Extract routes from OR-Tools solution"""
        routes = []
        
        for vehicle_id in range(num_vehicles):
            route_distance = 0
            route_load = 0
            route = []
            
            index = routing.Start(vehicle_id)
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)
            
            # Add the depot at the end
            route.append(manager.IndexToNode(index))
            
            # Only add non-empty routes
            if len(route) > 2:
                routes.append(route)
        
        return routes
    
    def _convert_to_original_indices(self, routes: List[List[int]], depots: np.ndarray, 
                                   customers: np.ndarray) -> List[List[int]]:
        """Convert OR-Tools indices back to original problem indices"""
        original_routes = []
        depot_idx = depots[0] if len(depots) > 0 else 0
        
        for route in routes:
            original_route = []
            for node in route:
                if node == 0:  # Depot
                    original_route.append(depot_idx)
                else:  # Customer
                    if node - 1 < len(customers):
                        original_route.append(customers[node - 1])
                    else:
                        original_route.append(depot_idx)  # Fallback
            original_routes.append(original_route)
        
        return original_routes
    
    def _solve_with_fallback(self, instance_idx: int, num_realizations: int) -> Dict:
        """Fallback to NN+2opt solver"""
        return self.fallback_solver.solve_instance(instance_idx, num_realizations)
    
    def _get_fallback_data(self) -> Dict:
        """Get empty data structure for fallback"""
        return {
            'distance_matrix': [[0]],
            'demands': [0],
            'vehicle_capacities': [100],
            'num_vehicles': 1,
            'num_locations': 1,
            'depot': 0
        }