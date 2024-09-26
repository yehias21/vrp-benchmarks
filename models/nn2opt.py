import multiprocessing as mp
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm

class TwoOptVRP:
    def __init__(self, distance_matrix, demands, vehicle_capacity, num_vehicles=None):
        self.distance_matrix = distance_matrix
        self.num_nodes = len(distance_matrix)
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles or self.estimate_num_vehicles()
        self.best_solution = None
        self.best_cost = float('inf')

    def estimate_num_vehicles(self):
        return int(np.ceil(sum(self.demands[1:]) / self.vehicle_capacity))

    def run(self):
        initial_solution = self.get_initial_solution()
        improved_solution = self.apply_two_opt(initial_solution)
        total_cost = self.evaluate_solution(improved_solution)
        self.best_solution = improved_solution
        self.best_cost = total_cost
        return self.best_solution, self.best_cost

    def get_initial_solution(self):
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
                    key=lambda node: self.distance_matrix[last_node][node]
                )
                route.append(next_node)
                capacity -= self.demands[next_node]
                unvisited.remove(next_node)
            route.append(0)  # Return to depot
            solution.append(route)
        return solution

    def apply_two_opt(self, solution):
        improved = True
        while improved:
            improved = False
            for route_idx, route in enumerate(solution):
                best_distance = self.route_distance(route)
                route_length = len(route)
                for i in range(1, route_length - 2):
                    for j in range(i + 1, route_length - 1):
                        if j - i == 1:  # Skip adjacent nodes to avoid unnecessary swaps
                            continue
                        new_route = route[:]
                        new_route[i:j] = route[j - 1:i - 1:-1]  # Perform 2-opt swap
                        if not self.is_route_feasible(new_route):
                            continue
                        new_distance = self.route_distance(new_route)
                        if new_distance < best_distance:
                            solution[route_idx] = new_route
                            improved = True
                            break  # Exit the loop over j
                    if improved:
                        break  # Exit the loop over i
                if improved:
                    break  # Exit the loop over routes
        return solution

    def route_distance(self, route):
        return sum(
            self.distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1)
        )

    def is_route_feasible(self, route):
        capacity = self.vehicle_capacity
        for node in route[1:-1]:  # Exclude depots at start and end
            capacity -= self.demands[node]
            if capacity < 0:
                return False
        return True

    def evaluate_solution(self, solution):
        total_distance = 0
        for route in solution:
            total_distance += self.route_distance(route)
        return total_distance


def solve_cvrp(instance):
    locs = instance["locs"]
    demands = instance["demands"]
    vehicle_capacity = instance["capacity"]

    # Build distance matrix
    SCALE_FACTOR = 1000
    num_locations = len(locs)
    distance_matrix = [[0] * num_locations for _ in range(num_locations)]
    for i in range(num_locations):
        for j in range(num_locations):
            distance_matrix[i][j] = int(np.linalg.norm(locs[i] - locs[j]) * SCALE_FACTOR)

    # Run 2-opt heuristic
    two_opt_vrp = TwoOptVRP(distance_matrix, demands, vehicle_capacity)
    best_solution, best_cost = two_opt_vrp.run()

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
                results.append((i, result))
                pbar.update(1)
                obj_distance = sum(r[1] for r in results)
                pbar.set_description(
                    f"Objective distance: {obj_distance:.2f} | Avg: {obj_distance / len(results):.4f}"
                )

    obj_distance = sum(r[1] for r in results)
    avg_obj_distance = obj_distance / num_instances
    print(f"Average objective distance: {avg_obj_distance:.4f}")


if __name__ == "__main__":
    main()
