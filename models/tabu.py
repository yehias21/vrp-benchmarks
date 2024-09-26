import multiprocessing as mp
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm

class TabuSearch:
    def __init__(
        self,
        distance_matrix,
        demands,
        vehicle_capacity,
        tabu_list_size=10,
        num_vehicles=None,
        neighborhood_size=100,
    ):
        self.distance_matrix = distance_matrix
        self.num_nodes = len(distance_matrix)
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.tabu_list_size = tabu_list_size
        self.tabu_list = []
        self.num_vehicles = num_vehicles or self.estimate_num_vehicles()
        self.neighborhood_size = neighborhood_size
        self.best_solution = None
        self.best_cost = float('inf')

    def estimate_num_vehicles(self):
        return int(np.ceil(sum(self.demands) / self.vehicle_capacity))

    def run(self, num_iterations=100):
        current_solution = self.get_initial_solution()
        current_cost = self.evaluate_solution(current_solution)
        self.best_solution = [route.copy() for route in current_solution]
        self.best_cost = current_cost

        for _ in range(num_iterations):
            neighbors = self.get_neighbors(current_solution)
            best_neighbor = None
            best_neighbor_cost = float('inf')

            for neighbor in neighbors:
                neighbor_key = tuple(tuple(route) for route in neighbor)
                if neighbor_key in self.tabu_list:
                    continue
                cost = self.evaluate_solution(neighbor)
                if cost < best_neighbor_cost:
                    best_neighbor = [route.copy() for route in neighbor]
                    best_neighbor_cost = cost

            if best_neighbor is None:
                break  # No better neighbor found

            current_solution = best_neighbor
            current_cost = best_neighbor_cost

            if current_cost < self.best_cost:
                self.best_solution = [route.copy() for route in current_solution]
                self.best_cost = current_cost

            self.update_tabu_list(current_solution)

        return self.best_solution, self.best_cost

    def get_initial_solution(self):
        # Simple nearest neighbor heuristic to get initial solution
        solution = []
        unvisited = set(range(1, self.num_nodes))
        for _ in range(self.num_vehicles):
            route = [0]
            capacity = self.vehicle_capacity
            while unvisited:
                last_node = route[-1]
                feasible_nodes = [
                    node for node in unvisited if self.demands[node] <= capacity
                ]
                if not feasible_nodes:
                    break
                next_node = min(
                    feasible_nodes,
                    key=lambda node: self.distance_matrix[last_node][node],
                )
                route.append(next_node)
                capacity -= self.demands[next_node]
                unvisited.remove(next_node)
            route.append(0)  # Return to depot
            solution.append(route)
        return solution

    def get_neighbors(self, solution):
        neighbors = []
        for _ in range(self.neighborhood_size):
            neighbor = self.swap_nodes(solution)
            neighbors.append(neighbor)
        return neighbors

    def swap_nodes(self, solution):
        neighbor = [route.copy() for route in solution]
        route_indices = list(range(len(neighbor)))
        if len(route_indices) < 2:
            return neighbor  # Can't swap between fewer than 2 routes
        r1, r2 = np.random.choice(route_indices, 2, replace=False)
        route1 = neighbor[r1]
        route2 = neighbor[r2]

        # Ensure there are nodes to swap (excluding depots)
        if len(route1) > 3 and len(route2) > 3:
            idx1 = np.random.randint(1, len(route1) - 1)
            idx2 = np.random.randint(1, len(route2) - 1)

            node1 = route1[idx1]
            node2 = route2[idx2]

            demand1 = self.demands[node1]
            demand2 = self.demands[node2]

            # Check capacity constraints
            route1_demand = sum(self.demands[node] for node in route1[1:-1]) - demand1 + demand2
            route2_demand = sum(self.demands[node] for node in route2[1:-1]) - demand2 + demand1

            if route1_demand <= self.vehicle_capacity and route2_demand <= self.vehicle_capacity:
                route1[idx1], route2[idx2] = node2, node1

        return neighbor

    def evaluate_solution(self, solution):
        total_distance = 0
        for route in solution:
            route_distance = sum(
                self.distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1)
            )
            total_distance += route_distance
        return total_distance

    def update_tabu_list(self, solution):
        solution_key = tuple(tuple(route) for route in solution)
        self.tabu_list.append(solution_key)
        if len(self.tabu_list) > self.tabu_list_size:
            self.tabu_list.pop(0)


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

    # Run Tabu Search
    tabu_search = TabuSearch(distance_matrix, demands, vehicle_capacity)
    best_solution, best_cost = tabu_search.run(num_iterations=100)

    return best_cost / SCALE_FACTOR  # Return distance scaled back


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
            "locs": np.vstack([dataset["depot"][i], dataset["locs"][i]]),
            "demands": np.hstack([[0], dataset["demand"][i]]),
            "capacity": dataset["capacity"][i],
        }
        for i in range(num_instances)
    ]

    with mp.Pool(processes=args.num_processes) as pool:
        results = []
        with tqdm(total=num_instances) as pbar:
            for i, result in pool.imap_unordered(solve_cvrp_wrapper, enumerate(instances)):
                results.append(result)
                pbar.update(1)
                obj_distance = sum(results)
                pbar.set_description(
                    f"Objective distance: {obj_distance:.2f} | Avg: {obj_distance / len(results):.4f}"
                )

    obj_distance = sum(results)
    avg_obj_distance = obj_distance / num_instances
    print(f"Average objective distance: {avg_obj_distance:.4f}")


if __name__ == "__main__":
    main()
