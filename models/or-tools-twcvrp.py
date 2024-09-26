import multiprocessing as mp
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def solve_twcvrp(data):
    manager = pywrapcp.RoutingIndexManager(
        len(data["time_matrix"]), data["num_vehicles"], data["depot"]
    )

    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["time_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    time = "Time"
    routing.AddDimension(
        evaluator_index=transit_callback_index,
        slack_max=30,  # allow waiting time
        capacity=1440,  # maximum time per vehicle
        fix_start_cumul_to_zero=False,  # Don't force start cumul to zero.
        name=time,
    )
    time_dimension = routing.GetDimensionOrDie(time)
    for location_idx, time_window in enumerate(data["time_windows"]):
        if location_idx == data["depot"]:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

    depot_idx = data["depot"]
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(
            data["time_windows"][depot_idx][0], data["time_windows"][depot_idx][1]
        )

    # Instantiate route start and end times to produce feasible times.
    for i in range(data["num_vehicles"]):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i))
        )
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))

    # this is to allow the model to drop some locations, otherwise it wouldn't produce a solution
    penalty = 1410 * len(data["time_matrix"])
    for node in range(1, len(data["time_matrix"])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
    )
    search_parameters.time_limit.FromSeconds(10)
    solution = routing.SolveWithParameters(search_parameters)

    total_route_time = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route_time = 0
        while not routing.IsEnd(index):
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_time += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        total_route_time += route_time
    return total_route_time


def solve_twcvrp_wrapper(args):
    i, instance = args
    result = solve_twcvrp(instance)
    return i, result


def get_time_matrix(num_customers, travel_times) -> list:
    matrix_keys = sorted(list(travel_times.keys()))
    time_matrix = []
    for k in matrix_keys:
        time_matrix.append(travel_times[k])
    time_matrix = np.array(time_matrix, dtype=np.uint16)
    time_matrix = time_matrix.reshape(num_customers, num_customers)
    return time_matrix.tolist()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, default="data/real_twcvrp/twvrp_10.npz"
    )
    parser.add_argument("--num_processes", "-n", type=int, default=2)
    args = parser.parse_args()
    dataset = dict(np.load(args.dataset_path, allow_pickle=True))
    num_instances = len(dataset["locations"])
    num_customers = len(dataset["locations"][0])
    instances = [
        {
            "depot": 0,
            "num_vehicles": 1,
            "locations": dataset["locations"][i],
            "demands": dataset["demands"][i],
            "time_matrix": get_time_matrix(num_customers, dataset["travel_times"][i]),
            "time_windows": dataset["time_windows"][i].tolist(),
        }
        for i in range(num_instances)
    ]

    with mp.Pool(processes=args.num_processes) as pool:
        results = []
        with tqdm(total=num_instances) as pbar:
            for i, result in pool.imap_unordered(
                solve_twcvrp_wrapper, enumerate(instances)
            ):
                results.append(result)
                pbar.update(1)
                obj_time = sum(results)
                pbar.set_description(
                    f"[{i}] Objective time: {obj_time:.2f} | Avg: {obj_time / (i+1):.4f}"
                )

    obj_time = sum(results)
    avg_obj_time = obj_time / num_instances
    print(f"Average objective time: {avg_obj_time:.4f}")


if __name__ == "__main__":
    main()
