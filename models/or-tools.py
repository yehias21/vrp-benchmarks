import multiprocessing as mp
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def solve_cvrp(instance):
    """Capacited Vehicles Routing Problem (CVRP).

    Args:
        instance (dict): A dictionary containing the following
            keys: depot, locs, demand, capacity.
    """
    SCALE_FACTOR = 1000
    locs = instance["locs"]

    data = {}
    data["distance_matrix"] = [[0] * len(locs) for _ in range(len(locs))]
    for i in range(len(locs)):
        for j in range(len(locs)):
            # we must convert the distances into integers, since OR-Tools uses
            # Mixed Integer programming and cannot deal with floats
            data["distance_matrix"][i][j] = int(
                np.linalg.norm(locs[i] - locs[j]) * SCALE_FACTOR
            )

    data["distance_matrix"] = data["distance_matrix"]
    data["demands"] = list(map(int, instance["demands"]))
    data["vehicle_capacities"] = [int(instance["capacity"])]
    data["num_vehicles"] = 1
    data["depot"] = 0
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        evaluator_index=demand_callback_index,
        slack_max=0,
        vehicle_capacities=data["vehicle_capacities"],
        fix_start_cumul_to_zero=True,
        name="Capacity",
    )
    # this is to allow the model to drop some locations, otherwise it wouldn't produce a solution
    penalty = 1000
    for node in range(1, len(data["distance_matrix"])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
    )
    search_parameters.time_limit.FromSeconds(1)
    solution = routing.SolveWithParameters(search_parameters)
    total_route_distance = 0  # distance of all routes without the penalty
    if not solution:
        return None
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        total_route_distance += route_distance
    return total_route_distance / SCALE_FACTOR


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
