import multiprocessing as mp
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm

class LKH3_VRP:
    def __init__(self, distance_matrix, demands, vehicle_capacity):
        self.distance_matrix = distance_matrix
        self.num_nodes = len(distance_matrix)
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.best_solution = None
        self.best_cost = float('inf')

    def run(self, max_iterations=100):
        initial_solution = self.get_initial_solution()
        self.best_solution = [route.copy() for route in initial_solution]
        self.best_cost = self.evaluate_solution(self.best_solution)

        for _ in range(max_iterations):
            improved = self.search()
            if not improved:
                break  # No further improvement

        return self.best_solution, self.best_cost

    def get_initial_solution(self):
        # Clarke and Wright Savings algorithm for initial solution
        routes = [[0, i, 0] for i in range(1, self.num_nodes)]
        savings = []
        for i in range(1, self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                saving = (
                    self.distance_matrix[0][i]
                    + self.distance_matrix[0][j]
                    - self.distance_matrix[i][j]
                )
                savings.append((saving, i, j))
        savings.sort(reverse=True)

        for saving, i, j in savings:
            route_i = None
            route_j = None
            for route in routes:
                if route[1] == i and len(route) == 3:
                    route_i = route
                if route[1] == j and len(route) == 3:
                    route_j = route
            if route_i and route_j and route_i != route_j:
                combined_demand = sum(self.demands[node] for node in route_i[1:-1] + route_j[1:-1])
                if combined_demand <= self.vehicle_capacity:
                    new_route = route_i[:-1] + route_j[1:]
                    routes.remove(route_i)
                    routes.remove(route_j)
                    routes.append(new_route + [0])  # Ensure route ends with depot
        return routes

    def search(self):
        improved = False
        # Inter-route and intra-route optimization
        for move_function in [self.two_opt_move, self.relocate_move, self.exchange_move]:
            improvement = True
            while improvement:
                improvement = False
                for route_i in self.best_solution:
                    for route_j in self.best_solution:
                        if move_function(route_i, route_j):
                            improvement = True
                            improved = True
                            break  # Restart after each improvement
                    if improvement:
                        break
                if improvement:
                    break  # Move to the next move type after an improvement
        return improved

    def two_opt_move(self, route_i, route_j):
        if route_i != route_j:
            return False  # 2-opt move is only intra-route
        best_gain = 0
        best_move = None
        route = route_i
        route_length = len(route)
        if route_length < 4:
            return False  # Not enough nodes to perform 2-opt
        for i in range(1, route_length - 2):
            for j in range(i + 1, route_length - 1):
                gain = self.calculate_2opt_gain(route, i, j)
                if gain < best_gain:
                    best_gain = gain
                    best_move = (i, j)
        if best_move:
            i, j = best_move
            route[i:j+1] = reversed(route[i:j+1])
            self.update_best_solution()
            return True
        return False

    def calculate_2opt_gain(self, route, i, j):
        a, b = route[i - 1], route[i]
        c, d = route[j], route[(j + 1) % len(route)]
        before = self.distance_matrix[a][b] + self.distance_matrix[c][d]
        after = self.distance_matrix[a][c] + self.distance_matrix[b][d]
        return after - before

    def relocate_move(self, route_i, route_j):
        improved = False
        if len(route_i) <= 3:
            return False  # Not enough nodes to relocate
        for i in range(1, len(route_i) - 1):
            node = route_i[i]
            demand = self.demands[node]
            if route_i == route_j:
                # Intra-route relocation
                for j in range(1, len(route_i) - 1):
                    if i == j:
                        continue
                    gain = self.calculate_relocate_gain(route_i, i, j)
                    if gain < 0:
                        route_i.pop(i)
                        if j >= i:
                            j -= 1  # Adjust index after removal
                        route_i.insert(j, node)
                        self.update_best_solution()
                        return True
            else:
                # Inter-route relocation
                if self.check_capacity(route_j, demand):
                    gain = self.calculate_inter_relocate_gain(route_i, route_j, i)
                    if gain < 0:
                        route_i.pop(i)
                        route_j.insert(-1, node)
                        self.update_best_solution()
                        return True
        return improved

    def calculate_relocate_gain(self, route, i, j):
        if i == j:
            return 0
        a, b, c = route[i - 1], route[i], route[i + 1]
        gain = self.distance_matrix[a][b] + self.distance_matrix[b][c] - self.distance_matrix[a][c]
        d, e = route[j - 1], route[j]
        gain += self.distance_matrix[d][e] - self.distance_matrix[d][b] - self.distance_matrix[b][e]
        return gain

    def calculate_inter_relocate_gain(self, route_i, route_j, i):
        a, b, c = route_i[i - 1], route_i[i], route_i[i + 1]
        cost_removed = self.distance_matrix[a][b] + self.distance_matrix[b][c] - self.distance_matrix[a][c]
        d, e = route_j[-2], route_j[-1]
        cost_added = self.distance_matrix[d][b] + self.distance_matrix[b][e] - self.distance_matrix[d][e]
        return cost_added - cost_removed

    def exchange_move(self, route_i, route_j):
        best_gain = 0
        best_move = None
        for i in range(1, len(route_i) - 1):
            for j in range(1, len(route_j) - 1):
                node_i = route_i[i]
                node_j = route_j[j]
                demand_i = self.demands[node_i]
                demand_j = self.demands[node_j]
                if self.check_capacity_swap(route_i, route_j, demand_i, demand_j):
                    gain = self.calculate_exchange_gain(route_i, route_j, i, j)
                    if gain < best_gain:
                        best_gain = gain
                        best_move = (i, j)
        if best_move:
            i, j = best_move
            route_i[i], route_j[j] = route_j[j], route_i[i]
            self.update_best_solution()
            return True
        return False

    def calculate_exchange_gain(self, route_i, route_j, i, j):
        a1, b1, c1 = route_i[i - 1], route_i[i], route_i[i + 1]
        a2, b2, c2 = route_j[j - 1], route_j[j], route_j[j + 1]
        before = (
            self.distance_matrix[a1][b1] + self.distance_matrix[b1][c1]
            + self.distance_matrix[a2][b2] + self.distance_matrix[b2][c2]
        )
        after = (
            self.distance_matrix[a1][b2] + self.distance_matrix[b2][c1]
            + self.distance_matrix[a2][b1] + self.distance_matrix[b1][c2]
        )
        return after - before

    def check_capacity(self, route, demand):
        total_demand = sum(self.demands[node] for node in route[1:-1]) + demand
        return total_demand <= self.vehicle_capacity

    def check_capacity_swap(self, route_i, route_j, demand_i, demand_j):
        total_demand_i = sum(self.demands[node] for node in route_i[1:-1]) - demand_i + demand_j
        total_demand_j = sum(self.demands[node] for node in route_j[1:-1]) - demand_j + demand_i
        return total_demand_i <= self.vehicle_capacity and total_demand_j <= self.vehicle_capacity

    def update_best_solution(self):
        cost = self.evaluate_solution(self.best_solution)
        if cost < self.best_cost:
            self.best_cost = cost
            # Deep copy to preserve the current best solution
            self.best_solution = [route.copy() for route in self.best_solution]

    def evaluate_solution(self, solution):
        total_distance = 0
        for route in solution:
            route_distance = sum(
                self.distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1)
            )
            total_distance += route_distance
        return total_distance

def solve_cvrp(instance):
    locs = instance["locs"]
    demands = instance["demands"]
    vehicle_capacity = instance["capacity"]

    # Build distance matrix
    SCALE_FACTOR = 1000
    num_locations = len(locs)
    distance_matrix = np.zeros((num_locations, num_locations), dtype=int)
    for i in range(num_locations):
        for j in range(num_locations):
            distance_matrix[i][j] = int(np.linalg.norm(locs[i] - locs[j]) * SCALE_FACTOR)

    # Run LKH3-like VRP solver
    lkh3_vrp = LKH3_VRP(distance_matrix, demands, vehicle_capacity)
    best_solution, best_cost = lkh3_vrp.run(max_iterations=100)

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
        manager = mp.Manager()
        lock = manager.Lock()
        results = []
        with tqdm(total=num_instances) as pbar:
            for i, result in pool.imap_unordered(solve_cvrp_wrapper, enumerate(instances)):
                results.append((i, result))
                with lock:
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
