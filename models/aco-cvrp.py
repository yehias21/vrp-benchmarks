import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Dict, List
import sys

# Ensure that the parent directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vrp_bench.real_cvrp import generate_cvrp_dataset
from vrp_bench.common import load_dataset

# ACO Parameters
NUM_ANTS = 10
NUM_ITERATIONS = 100
ALPHA = 1.0           # Influence of pheromone on direction
BETA = 2.0            # Influence of heuristic (inverse of distance)
EVAPORATION_RATE = 0.5  # Rate at which pheromone evaporates

def get_distance_matrix(locations):
    """
    Compute the Euclidean distance matrix for a set of locations.

    Parameters:
    locations (np.ndarray): A 2D array of shape (n, 2), where n is the number of locations, and each location is given as [x, y].

    Returns:
    list: A 2D list representing the distance matrix, where each entry [i][j] is the distance from location i to location j.
    """
    num_locations = len(locations)
    distance_matrix = np.zeros((num_locations, num_locations))

    for i in range(num_locations):
        for j in range(num_locations):
            # Compute the Euclidean distance between locations i and j
            distance_matrix[i, j] = np.linalg.norm(locations[i] - locations[j])

    # Convert to a list of lists for compatibility with non-numpy environments
    return distance_matrix.tolist()

def solve_svrp_wrapper(args):
    i, instance, num_realizations = args
    # Generate realizations within the wrapper function to avoid multiprocessing issues
    realizations = [get_distance_matrix(instance["locations"]) for _ in range(num_realizations)]
    # Attach realizations to the instance data in a simplified way
    instance_with_realizations = {
        **instance,
        "realizations": realizations
    }
    aco = AntColonyOptimizer(instance_with_realizations, num_realizations)
    best_route, best_route_length = aco.run()
    return i, best_route_length

class AntColonyOptimizer:
    def __init__(self, data: Dict, num_realizations: int):
        self.data = data
        self.num_realizations = num_realizations
        self.num_locations = len(data["distance_matrix"])
        self.pheromone_matrix = np.ones((self.num_locations, self.num_locations))

    def heuristic_matrix(self):
        # Inverse of the distance matrix for heuristic information
        distance_matrix = np.array(self.data["distance_matrix"])
        return 1 / (distance_matrix + 1e-10)

    def select_next_location(self, current_loc, unvisited, heuristic):
        pheromones = self.pheromone_matrix[current_loc][unvisited]
        heuristic_values = heuristic[current_loc][unvisited]
        probs = (pheromones ** ALPHA) * (heuristic_values ** BETA)
        probs /= probs.sum()  # Normalize to make it a probability distribution
        next_loc = np.random.choice(unvisited, p=probs)
        return next_loc

    def update_pheromones(self, routes: List[List[int]], avg_route_length: float):
        self.pheromone_matrix *= (1 - EVAPORATION_RATE)  # Evaporation
        for route in routes:
            for i in range(len(route) - 1):
                from_loc, to_loc = route[i], route[i + 1]
                self.pheromone_matrix[from_loc][to_loc] += 1.0 / avg_route_length  # Deposit pheromone

    def run(self):
        best_route_length = float("inf")
        best_route = None
        heuristic = self.heuristic_matrix()
        
        for _ in range(NUM_ITERATIONS):
            routes = []
            route_lengths = []
            for _ in range(NUM_ANTS):
                route = [0]  # Start from depot (assuming index 0)
                unvisited = list(range(1, self.num_locations))  # Exclude depot
                while unvisited:
                    current_loc = route[-1]
                    next_loc = self.select_next_location(current_loc, unvisited, heuristic)
                    route.append(next_loc)
                    unvisited.remove(next_loc)
                route.append(0)  # Return to depot
                routes.append(route)
                
                # Calculate route length over all realizations
                realization_lengths = [
                    sum(self.data["realizations"][r][route[i]][route[i + 1]]
                        for i in range(len(route) - 1))
                    for r in range(self.num_realizations)
                ]
                avg_route_length = np.mean(realization_lengths)
                route_lengths.append(avg_route_length)

                # Update best solution
                if avg_route_length < best_route_length:
                    best_route_length = avg_route_length
                    best_route = route

            # Update pheromones based on average route length across realizations
            self.update_pheromones(routes, best_route_length)
        
        return best_route, best_route_length

def main():
    parser = ArgumentParser()
    parser.add_argument("--num_customers", type=int, default=10)
    parser.add_argument("--num_training_instances", type=int, default=5)
    parser.add_argument("--num_realizations", type=int, default=3, help="Number of realizations per rl instance")
    parser.add_argument("--num_processes", "-n", type=int, default=2)
    args = parser.parse_args()

    # Generate Training Data (not saved)
    train_instances = []
    for _ in range(args.num_training_instances):
        base_instance = generate_cvrp_dataset(num_customers=args.num_customers)
        instance = {
            "depot": 0,
            "locations": base_instance["locations"][0],
            "demands": base_instance["demands"][0],
            "num_vehicles": 1,
            "vehicle_capacities": [1000],
            "distance_matrix": get_distance_matrix(base_instance["locations"][0]),
        }
        train_instances.append(instance)

    # Parallel rl
    with mp.Pool(processes=args.num_processes) as pool:
        results = []
        with tqdm(total=args.num_training_instances) as pbar:
            for i, result in pool.imap_unordered(
                solve_svrp_wrapper, [(i, instance, args.num_realizations) for i, instance in enumerate(train_instances)]
            ):
                results.append(result)
                pbar.update(1)

    # Load and prepare test data
    test_data_path = f"data/real_cvrp/cvrp_{args.num_customers}.npz"
    test_data = dict(np.load(test_data_path, allow_pickle=True))
    num_test_instances = len(test_data["locations"])

    test_instances = [
        {
            "depot": 0,
            "locations": test_data["locations"][i],
            "demands": test_data["demands"][i],
            "num_vehicles": test_data["num_vehicles"][i].item(),
            "vehicle_capacities": [
                test_data["vehicle_capacities"][i].item()
                for _ in range(test_data["num_vehicles"][i])
            ],
            "distance_matrix": get_distance_matrix(test_data["locations"][i]),
        }
        for i in range(num_test_instances)
    ]

    # Parallel testing
    with mp.Pool(processes=args.num_processes) as pool:
        test_results = []
        with tqdm(total=num_test_instances) as pbar:
            for i, result in pool.imap_unordered(
                solve_svrp_wrapper, [(i, instance, 1) for i, instance in enumerate(test_instances)]
            ):
                test_results.append(result)
                pbar.update(1)

    avg_train_distance = sum(results) / len(results)
    avg_test_distance = sum(test_results) / len(test_results)

    print(f"Average rl distance: {avg_train_distance:.4f}")
    print(f"Average test distance: {avg_test_distance:.4f}")

if __name__ == "__main__":
    main()
