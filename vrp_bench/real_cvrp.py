import os
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from common import (
    generate_base_instance,
    save_dataset,
    load_dataset,
    visualize_instance,
)
from constants import NUM_INSTANCES, DEMAND_RANGE, MAP_SIZE

CAPACITIES = {
    10: 20.0,
    15: 25.0,
    20: 30.0,
    30: 33.0,
    40: 37.0,
    50: 40.0,
    60: 43.0,
    75: 45.0,
    100: 50.0,
    125: 55.0,
    150: 60.0,
    200: 70.0,
    500: 100.0,
    1000: 150.0,
}

def generate_cvrp_instance(
        num_customers: int,
        num_cities: Optional[int] = None,
        num_depots: int = 3,  # default now allows multi-depots (set to 3)
        is_dynamic: bool = False,
) -> Dict:
    """Generate a base CVRP instance with support for multiple depots."""
    num_cities = num_cities if num_cities else max(1, num_customers // 50)
    instance = generate_base_instance(
        num_customers, MAP_SIZE, num_cities, num_depots, DEMAND_RANGE, is_dynamic
    )
    return instance


def get_num_vehicles(instance) -> int:
    """
    Compute the number of vehicles needed for the instance.
    A simple heuristic is to take the total customer demand, then divide by the vehicle's capacity.
    Here we ensure that we always have at least one vehicle.
    """
    total_demand = np.sum(instance["demands"])
    vehicle_capacity = instance["vehicle_capacity"]
    # Calculate the minimum vehicles needed (rounding up)
    num_vehicles = max(1, int(np.ceil(total_demand / vehicle_capacity)))
    return num_vehicles


def generate_cvrp_dataset(
        num_customers: int,
        num_cities: Optional[int] = None,
        num_depots: int = 1,
        precision=np.uint16,
        is_dynamic: bool = False,
) -> Dict:
    """Generate a CVRP dataset with dynamic multi-vehicle and multi-depot support."""
    num_cities = num_cities if num_cities else max(1, num_customers // 50)
    dataset = {
        "locations": [],
        "demands": [],
        "num_vehicles": [],
        "vehicle_capacities": [],
        "appear_times": [],
        "map_size": MAP_SIZE,
        "num_cities": num_cities,
        "num_depots": num_depots,
    }

    for _ in tqdm(
            range(NUM_INSTANCES), desc=f"Generating {num_customers} customer instances"
    ):
        # Generate a new instance with the given multi-depot configuration
        instance = generate_cvrp_instance(
            num_customers,
            num_cities,
            num_depots,
            is_dynamic=is_dynamic,
        )

        # Dynamic number of vehicles computed via a capacity-based heuristic:
        num_vehicles = 1 #get_num_vehicles(instance)

        dataset["locations"].append(instance["locations"].astype(precision))
        dataset["demands"].append(instance["demands"].astype(precision))
        # Allocate the full capacity for each computed vehicle
        # dataset["vehicle_capacities"].append([instance["vehicle_capacity"]]) # * num_vehicles)
        dataset["vehicle_capacities"].append([CAPACITIES[num_customers]] )
        dataset["appear_times"].append(instance["appear_time"])
        dataset["num_vehicles"].append(num_vehicles)

    return {k: np.array(v, dtype=object) for k, v in dataset.items()}


def main():

    customer_counts = [10, 20, 50, 100, 200, 500, 1000]
    os.makedirs("../data/real_cvrp", exist_ok=True)
    for num_customers in tqdm(customer_counts):
        depots = max(1, num_customers // 50)
        dataset = generate_cvrp_dataset(num_customers, num_depots=1)
        save_dataset(dataset, f"../data/real_cvrp/cvrp_{num_customers}_single_depot_single_vehicule_capacities.npz")


if __name__ == "__main__":
    main()
    # dataset = load_dataset("../data/real_cvrp/cvrp_1000.npz")
    # visualize_instance(dataset)
