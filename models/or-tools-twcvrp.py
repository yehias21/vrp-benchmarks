import multiprocessing as mp
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def solve_twcvrp(data):
    # For multi-depot: if "vehicle_depots" is provided, use those as start and end nodes.
    if "vehicle_depots" in data:
        vehicle_depots = data["vehicle_depots"]
        manager = pywrapcp.RoutingIndexManager(
            len(data["time_matrix"]), data["num_vehicles"], vehicle_depots, vehicle_depots
        )
        # Prepare the list of depot nodes for later use.
        depot_nodes = set(vehicle_depots)
    else:
        manager = pywrapcp.RoutingIndexManager(
            len(data["time_matrix"]), data["num_vehicles"], data["depot"]
        )
        depot_nodes = {data["depot"]}

    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["time_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    time_label = "Time"
    routing.AddDimension(
        evaluator_index=transit_callback_index,
        slack_max=30,  # allow waiting time
        capacity=1440,  # maximum time per vehicle
        fix_start_cumul_to_zero=False,  # Don't force start cumul to zero.
        name=time_label,
    )
    time_dimension = routing.GetDimensionOrDie(time_label)

    # Set time windows for all locations except depots.
    for location_idx, time_window in enumerate(data["time_windows"]):
        if location_idx in depot_nodes:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

    # For each vehicle, set the time window constraint at its starting depot.
    for vehicle_id in range(data["num_vehicles"]):
        start_index = routing.Start(vehicle_id)
        depot_node = manager.IndexToNode(start_index)
        time_dimension.CumulVar(start_index).SetRange(
            data["time_windows"][depot_node][0], data["time_windows"][depot_node][1]
        )

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

    # Finalize route start and end times.
    for vehicle_id in range(data["num_vehicles"]):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(vehicle_id))
        )
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(vehicle_id))
        )

    # Allow dropping non-depot locations.
    penalty = 5000 * len(data["time_matrix"])
    for node in range(len(data["time_matrix"])):
        if node in depot_nodes:
            continue  # do not drop depot nodes.
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
    visited_clients = 0
    # Iterate each vehicle's route, count the visited clients (non-depot nodes)
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route_time = 0
        while not routing.IsEnd(index):
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_time += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            node = manager.IndexToNode(index)
            if node not in depot_nodes:
                visited_clients += 1
        total_route_time += route_time

    print(f"Visited clients in instance: {visited_clients}")
    return total_route_time, visited_clients


def solve_twcvrp_wrapper(args):
    i, instance = args
    route_time, visited = solve_twcvrp(instance)
    return i, route_time, visited


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
        "--dataset_path", type=str, default="../data/real_twcvrp/twvrp_1000.npz"
    )
    parser.add_argument("--num_processes", "-n", type=int, default=2)
    args = parser.parse_args()
    dataset = dict(np.load(args.dataset_path, allow_pickle=True))

    num_instances = len(dataset["locations"])
    num_customers = len(dataset["locations"][0])

    # For multi-depot, assume that dataset["num_depots"] holds the number of depots.
    # Here we assign each vehicle a depot in a round-robin fashion.
    num_depots = int(dataset["num_depots"])
    num_vehicles = int(dataset["num_vehicles"])
    vehicle_depots = [i % num_depots for i in range(num_vehicles)]

    instances = [
        {
            "vehicle_depots": vehicle_depots,
            "num_vehicles": num_vehicles,
            "locations": dataset["locations"][i],
            "demands": dataset["demands"][i],
            "time_matrix": get_time_matrix(num_customers, dataset["travel_times"][i]),
            "time_windows": list(map(lambda x: [0, 1440], dataset["time_windows"][i].tolist())),
            "vehicle_capacities": dataset["vehicle_capacities"][i]
        }
        for i in range(num_instances)
    ]

    total_route_time = 0
    total_visited_clients = 0
    with mp.Pool(processes=args.num_processes) as pool:
        with tqdm(total=num_instances) as pbar:
            for i, route_time, visited in pool.imap_unordered(solve_twcvrp_wrapper, enumerate(instances)):
                total_route_time += route_time
                total_visited_clients += visited
                avg_route_time = total_route_time / (i + 1)
                avg_visited = total_visited_clients / (i + 1)
                pbar.update(1)
                pbar.set_description(
                    f"[Instance {i}] Obj time total: {total_route_time:.2f} | Avg time: {avg_route_time:.4f} | "
                    f"Total visited: {total_visited_clients} | Avg visited: {avg_visited:.2f}"
                )

    avg_obj_time = total_route_time / num_instances
    print(f"Average objective time: {avg_obj_time:.4f}")
    print(f"Total visited customers across all instances: {total_visited_clients}")


if __name__ == "__main__":
    main()

