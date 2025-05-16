import multiprocessing as mp
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def solve_cvrp(data):
    # Determine depot assignment based on multi-depot support.
    if "vehicle_depots" in data:
        vehicle_depots = data["vehicle_depots"]
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], vehicle_depots, vehicle_depots
        )
        depot_set = set([i for i in range(data["num_depots"])])
    else:
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )
        depot_set = {data["depot"]}

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

    # Allow dropping non-depot nodes. Depot nodes are not droppable.
    penalty = 141000 * len(data["distance_matrix"])
    for node in range(len(data["distance_matrix"])):
        if node in depot_set:
            continue  # Skip depots.
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(10)
    solution = routing.SolveWithParameters(search_parameters)
    total_route_distance = 0  # Total distance without penalties

    if not solution:
        return None

    visited_customers = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            current_node = manager.IndexToNode(index)
            # Only count the node if it is not in the depot set
            if current_node not in depot_set:
                visited_customers += 1
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        total_route_distance += route_distance

    print(f"Visited customers: {visited_customers}")
    return total_route_distance


def solve_cvrp_wrapper(args):
    i, instance = args
    result = solve_cvrp(instance)
    return i, result


def get_distance_matrix(locations):
    num_locations = len(locations)
    distance_matrix = np.zeros((num_locations, num_locations))
    for i in range(num_locations):
        for j in range(num_locations):
            distance_matrix[i, j] = np.linalg.norm(locations[i] - locations[j])
    distance_matrix = distance_matrix.astype(np.uint16)
    return distance_matrix.tolist()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, default="../data/real_cvrp/cvrp_10_single_depot_single_vehicule_capacities.npz"
    )
    parser.add_argument("--num_processes", "-n", type=int, default=2)
    args = parser.parse_args()
    dataset = dict(np.load(args.dataset_path, allow_pickle=True))
    num_instances = len(dataset["locations"])

    # Check if the dataset provides the number of depots.
    # If so, assign vehicles to depots via a round-robin strategy.
    if "num_depots" in dataset:
        num_depots = int(dataset["num_depots"])
    else:
        num_depots = 1

    instances = []
    for i in range(num_instances):
        num_vehicles = int(dataset["num_vehicles"])
        # Generate vehicle depot assignment:
        if num_depots > 1:
            vehicle_depots = [j % num_depots for j in range(num_vehicles)]
        else:
            vehicle_depots = [0] * num_vehicles

        instance = {
            "locations": dataset["locations"][i],
            "demands": dataset["demands"][i],
            "num_vehicles": num_vehicles,
            "vehicle_capacities": [
                dataset["vehicle_capacities"][i].item() for _ in range(num_vehicles)
            ],
            "distance_matrix": get_distance_matrix(dataset["locations"][i]),
        }
        # For multi-depot, add the vehicle depot information. Otherwise, set a single depot.
        if num_depots > 1:
            instance["vehicle_depots"] = vehicle_depots
        else:
            instance["depot"] = 0
        instance["num_depots"] = num_depots

        instances.append(instance)

    with mp.Pool(processes=args.num_processes) as pool:
        results = []
        with tqdm(total=num_instances) as pbar:
            for i, result in pool.imap_unordered(solve_cvrp_wrapper, enumerate(instances)):
                results.append(result)
                pbar.update(1)
                obj_distance = sum(results)
                pbar.set_description(
                    f"[{i}] Objective distance: {obj_distance:.2f} | Avg: {obj_distance / (i + 1):.4f}"
                )

    obj_distance = sum(results)
    avg_obj_distance = obj_distance / num_instances
    print(f"Average objective distance: {avg_obj_distance:.4f}")


if __name__ == "__main__":
    main()
