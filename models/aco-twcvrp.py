import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Dict, List
import sys

# Ensure that the parent directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vrp_bench.real_twcvrp import generate_twcvrp_dataset  # Assuming similar to generate_cvrp_dataset
from vrp_bench.common import load_dataset

# ACO Parameters
NUM_ANTS = 10
NUM_ITERATIONS = 100
ALPHA = 1.0           # Influence of pheromone on direction
BETA = 2.0            # Influence of heuristic (inverse of distance)
EVAPORATION_RATE = 0.5  # Rate at which pheromone evaporates

def get_distance_matrix(locations):
    num_locations = len(locations)
    distance_matrix = np.zeros((num_locations, num_locations))
    for i in range(num_locations):
        for j in range(num_locations):
            distance_matrix[i, j] = np.linalg.norm(locations[i] - locations[j])
    return distance_matrix.tolist()

def solve_twcvrp_wrapper(args):
    i, instance, num_realizations = args
    realizations = [get_distance_matrix(instance["locations"]) for _ in range(num_realizations)]
    instance_with_realizations = {**instance, "realizations": realizations}
    aco = TWCVRP_AntColonyOptimizer(instance_with_realizations, num_realizations)
    best_route, best_route_length = aco.run()
    return i, best_route_length

class TWCVRP_AntColonyOptimizer:
    def __init__(self, data: Dict, num_realizations: int):
        self.data = data
        self.num_realizations = num_realizations
        self.num_locations = len(data["distance_matrix"])
        self.pheromone_matrix = np.ones((self.num_locations, self.num_locations))

    def heuristic_matrix(self):
        distance_matrix = np.array(self.data["distance_matrix"])
        return 1 / (distance_matrix + 1e-10)

    def select_next_location(self, current_loc, unvisited, heuristic, time_remaining):
        # Get pheromone and heuristic values only for unvisited locations
        pheromones = self.pheromone_matrix[current_loc][unvisited]
        heuristic_values = heuristic[current_loc][unvisited]
        
        # Calculate probabilities for the next location
        probs = (pheromones ** ALPHA) * (heuristic_values ** BETA)
        if probs.sum() == 0:
            return None  # No valid selection
        probs /= probs.sum()  # Normalize to make it a probability distribution

        # Temporarily remove time window constraints to debug feasibility
        next_loc = np.random.choice(unvisited, p=probs)
        return next_loc
        
    def update_pheromones(self, routes: List[List[int]], avg_route_length: float):
        self.pheromone_matrix *= (1 - EVAPORATION_RATE)
        
        if avg_route_length > 0:
            for route in routes:
                for i in range(len(route) - 1):
                    from_loc, to_loc = route[i], route[i + 1]
                    self.pheromone_matrix[from_loc][to_loc] += 1.0 / avg_route_length

    def run(self):
        best_route_length = float("inf")
        best_route = None
        heuristic = self.heuristic_matrix()
        
        for _ in range(NUM_ITERATIONS):
            routes = []
            route_lengths = []
            for _ in range(NUM_ANTS):
                route = [0]  # Start from depot
                unvisited = list(range(1, self.num_locations))
                # Set current_time to align closer to the first available customer
                earliest_customer_time = min(self.data["time_windows"][i][0] for i in unvisited)
                current_time = max(0, earliest_customer_time - 10)  # Start slightly earlier

                while unvisited:
                    current_loc = route[-1]
                    next_loc = self.select_next_location(current_loc, unvisited, heuristic, current_time)
                    if next_loc is None:
                        break
                    route.append(next_loc)
                    unvisited.remove(next_loc)
                    
                    # Calculate travel time and update current time
                    travel_time = heuristic[current_loc][next_loc]
                    current_time += travel_time
                    current_time = max(current_time, self.data["time_windows"][next_loc][0])

                route.append(0)  # Return to depot if no more locations can be visited
                routes.append(route)

                # Calculate realization lengths
                realization_lengths = []
                for r in range(self.num_realizations):
                    total_distance = 0
                    for i in range(len(route) - 1):
                        from_loc = route[i]
                        to_loc = route[i + 1]
                        distance = self.data["realizations"][r][from_loc][to_loc]
                        total_distance += distance
                    realization_lengths.append(total_distance)

                avg_route_length = np.mean(realization_lengths)

                route_lengths.append(avg_route_length)

                if avg_route_length < best_route_length:
                    best_route_length = avg_route_length
                    best_route = route

            self.update_pheromones(routes, best_route_length)
        
        return best_route, best_route_length

def main():
    parser = ArgumentParser()
    parser.add_argument("--num_customers", type=int, default=10)
    parser.add_argument("--num_training_instances", type=int, default=5)
    parser.add_argument("--num_realizations", type=int, default=3)
    parser.add_argument("--num_processes", "-n", type=int, default=2)
    args = parser.parse_args()

    # Training Data Generation and Evaluation
    train_instances = []
    for _ in range(args.num_training_instances):
        base_instance = generate_twcvrp_dataset(num_customers=args.num_customers)
        instance = {
            "depot": 0,
            "locations": base_instance["locations"][0],
            "demands": base_instance["demands"][0],
            "num_vehicles": 1,
            "vehicle_capacities": [1000],
            "distance_matrix": get_distance_matrix(base_instance["locations"][0]),
            "time_windows": base_instance["time_windows"][0]
        }
        train_instances.append(instance)

    with mp.Pool(processes=args.num_processes) as pool:
        train_results = []
        with tqdm(total=args.num_training_instances, desc="Training Instances") as pbar:
            for i, result in pool.imap_unordered(
                solve_twcvrp_wrapper, [(i, instance, args.num_realizations) for i, instance in enumerate(train_instances)]
            ):
                train_results.append(result)
                pbar.update(1)

    avg_train_distance = sum(train_results) / len(train_results)

    # Test Data Loading and Evaluation
    test_data_path = f"data/real_twcvrp/twvrp_{args.num_customers}.npz"
    test_data = dict(np.load(test_data_path, allow_pickle=True))
    num_test_instances = len(test_data["locations"])

    test_instances = [
        {
            "depot": 0,
            "locations": test_data["locations"][i],
            "demands": test_data["demands"][i],
            "num_vehicles": 1,
            "vehicle_capacities": [1000],
            "distance_matrix": get_distance_matrix(test_data["locations"][i]),
            "time_windows": test_data["time_windows"][i].tolist()
        }
        for i in range(num_test_instances)
    ]

    with mp.Pool(processes=args.num_processes) as pool:
        test_results = []
        with tqdm(total=num_test_instances, desc="Testing Instances") as pbar:
            for i, result in pool.imap_unordered(
                solve_twcvrp_wrapper, [(i, instance, 1) for i, instance in enumerate(test_instances)]
            ):
                test_results.append(result)
                pbar.update(1)

    avg_test_distance = sum(test_results) / len(test_results)
    print(f"Average rl_models distance: {avg_train_distance:.4f}")
    print(f"Average testing distance: {avg_test_distance:.4f}")


if __name__ == "__main__":
    main()
