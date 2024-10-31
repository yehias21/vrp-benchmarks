import os
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from .common import (
    generate_base_instance,
    save_dataset,
    load_dataset,
    visualize_instance,
)
from .constants import NUM_INSTANCES, DEMAND_RANGE, MAP_SIZE

def generate_cvrp_instance(
    num_customers: int,
    num_cities: Optional[int] = None,
    num_depots: int = 1,
    is_dynamic: bool = False,
) -> Dict:
    num_cities = num_cities if num_cities else max(1, num_customers // 50)
    instance = generate_base_instance(
        num_customers, MAP_SIZE, num_cities, num_depots, DEMAND_RANGE, is_dynamic
    )
    return instance


def generate_cvrp_dataset(
    num_customers: int,
    num_cities: Optional[int] = None,
    num_depots: int = 1,
    num_vehicles: int = 1,
    precision=np.uint16,
    is_dynamic: bool = False,
) -> Dict:
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
        instance = generate_cvrp_instance(
            num_customers,
            num_cities,
            num_depots,
            is_dynamic=is_dynamic,
        )
        dataset["locations"].append(instance["locations"].astype(precision))
        dataset["demands"].append(instance["demands"].astype(precision))
        dataset["vehicle_capacities"].append(
            [instance["vehicle_capacity"]] * num_vehicles
        )
        dataset["appear_times"].append(instance["appear_time"])
        dataset["num_vehicles"].append(1)

    return {k: np.array(v) for k, v in dataset.items()}


def main():
    customer_counts = [10, 20, 50, 100, 200, 500, 1000]
    os.makedirs("data/real_cvrp", exist_ok=True)

    for num_customers in tqdm(customer_counts):
        dataset = generate_cvrp_dataset(num_customers)
        save_dataset(dataset, f"data/real_cvrp/cvrp_{num_customers}.npz")


if __name__ == "__main__":
    main()
    dataset = load_dataset("data/real_cvrp/cvrp_10.npz")
    visualize_instance(dataset)
