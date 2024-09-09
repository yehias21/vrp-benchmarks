import multiprocessing as mp
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm

class AntColony:
    def __init__(self, distance_matrix, demands, vehicle_capacity, num_ants=10, alpha=1, beta=2, evaporation_rate=0.5, pheromone_deposit=1):
        self.distance_matrix = distance_matrix
        self.num_nodes = len(distance_matrix)
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        self.pheromone_matrix = np.ones((self.num_nodes, self.num_nodes))
        self.best_route = None
        self.best_distance = float('inf')

    def run(self, num_iterations=100):
        for _ in range(num_iterations):
            all_routes = []
            all_distances = []
            for _ in range(self.num_ants):
                route, route_distance = self.construct_solution()
                all_routes.append(route)
                all_distances.append(route_distance)
                if route_distance < self.best_distance:
                    self.best_distance = route_distance
                    self.best_route = route

            self.update_pheromones(all_routes, all_distances)

        return self.best_route, self.best_distance

    def construct_solution(self):
        visited = [False] * self.num_nodes
        current_capacity = self.vehicle_capacity
        route = [0]  # Start at depot
        current_node = 0
        route_distance = 0

        while len(route) < self.num_nodes:
            next_node = self.select_next_node(current_node, visited, current_capacity)
            if next_node is None:
                # Return to depot if no valid node is found
                route.append(0)
                route_distance += self.distance_matrix[current_node][0]
                current_capacity = self.vehicle_capacity
                current_node = 0
            else:
                route.append(next_node)
                visited[next_node] = True
                current_capacity -= self.demands[next_node]
                route_distance += self.distance_matrix[current_node][next_node]
                current_node = next_node

        route.append(0)  # Return to depot at the end
        route_distance += self.distance_matrix[current_node][0]

        return route, route_distance

    def select_next_node(self, current_node, visited, current_capacity):
        probabilities = []
        total_pheromone = 0
        for i in range(self.num_nodes):
            if not visited[i] and self.demands[i] <= current_capacity:
                pheromone = self.pheromone_matrix[current_node][i] ** self.alpha
                distance = self.distance_matrix[current_node][i] ** self.beta
                # Avoid division by zero and negative distances
                if distance > 0:
                    total_pheromone += pheromone / distance
                    probabilities.append(pheromone / distance)
                else:
                    probabilities.append(0)
            else:
                probabilities.append(0)

        if total_pheromone == 0:
            return None

        probabilities = [p / total_pheromone if total_pheromone > 0 else 0 for p in probabilities]
        
        # Ensure no NaN probabilities
        probabilities = np.nan_to_num(probabilities, nan=0.0)

        return np.random.choice(range(self.num_nodes), p=probabilities)

    def update_pheromones(self, routes, distances):
        self.pheromone_matrix *= (1 - self.evaporation_rate)  # Evaporation
        for route, distance in zip(routes, distances):
            for i in range(len(route) - 1):
                self.pheromone_matrix[route[i]][route[i + 1]] += self.pheromone_deposit / distance


def solve_cvrp(instance):
    locs = instance["locs"]
    demands = instance["demands"]
    vehicle_capacity = instance["capacity"]

    # Build distance matrix
    SCALE_FACTOR = 1000
    distance_matrix = [[0] * len(locs) for _ in range(len(locs))]
    for i in range(len(locs)):
        for j in range(len(locs)):
            distance_matrix[i][j] = int(np.linalg.norm(locs[i] - locs[j]) * SCALE_FACTOR)

    # Run Ant Colony Optimization
    colony = AntColony(distance_matrix, demands, vehicle_capacity, num_ants=10)
    best_route, best_distance = colony.run(num_iterations=100)

    return best_distance / SCALE_FACTOR  # return distance scaled back


def solve_cvrp_wrapper(args):
    i, instance = args
    result = solve_cvrp(instance)
    return i, result


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/cvrp/vrp_10_1000.npz")
    parser.add_argument("--num_processes", "-n", type=int, default=2)
    args = parser.parse_args()
    dataset = np.load(args.dataset_path)
    num_instances = len(dataset["locs"])

    instances = [
        {
            "depot": dataset["depot"][i],
            "locs": dataset["locs"][i],
            "demands": dataset["demand"][i],
            "capacity": dataset["capacity"][i],
        }
        for i in range(num_instances)
    ]
    with mp.Pool(processes=args.num_processes) as pool:
        results = []
        with tqdm(total=num_instances) as pbar:
            for i, result in pool.imap_unordered(
                solve_cvrp_wrapper, enumerate(instances)
            ):
                results.append(result)
                pbar.update(1)
                obj_distance = sum(results)
                pbar.set_description(
                    f"[{i}] Objective distance: {obj_distance:.2f} | Avg: {obj_distance / (i+1):.4f}"
                )

    obj_distance = sum(results)
    avg_obj_distance = obj_distance / num_instances
    print(f"Average objective distance: {avg_obj_distance:.4f}")


if __name__ == "__main__":
    main()
