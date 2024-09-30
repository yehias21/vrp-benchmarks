import os
from typing import Dict, Tuple, Optional
import random

import numpy as np
from tqdm import tqdm

from travel_time_generator import sample_travel_time, get_distances
from common import (
    generate_base_instance,
    save_dataset,
    load_dataset,
    visualize_instance,
)
from time_windows_generator import sample_time_window
from constants import NUM_INSTANCES, DEMAND_RANGE, MAP_SIZE


def generate_time_window(customer_appear_time: int) -> Tuple[int, int]:
    return sample_time_window(random.randint(0, 1), customer_appear_time)


def generate_twcvrp_instance(
    num_customers: int,
    num_cities: Optional[int] = None,
    num_depots: int = 1,
    is_dynamic: bool = False,
) -> Dict:
    num_cities = num_cities if num_cities else max(1, num_customers // 50)
    instance = generate_base_instance(
        num_customers, MAP_SIZE, num_cities, num_depots, DEMAND_RANGE, is_dynamic
    )
    distances = get_distances(instance["map_instance"])
    travel_times = {}
    for i in range(len(instance["locations"])):
        for j in range(len(instance["locations"])):
            if i != j:
                current_time = random.randint(
                    0, 1440
                )  # Random start time for each trip
                travel_times[(i, j)] = sample_travel_time(i, j, distances, current_time)
            else:
                travel_times[(i, j)] = 0

    time_windows = [
        generate_time_window(instance["appear_time"][i])
        for i in range(num_customers + num_depots)
    ]
    time_windows[0] = (0, 1440)  # Depot has no time window
    instance["travel_times"] = travel_times
    instance["time_windows"] = np.array(time_windows)

    return instance


def get_time_matrix(num_customers, travel_times) -> list:
    matrix_keys = sorted(list(travel_times.keys()))
    time_matrix = []
    for k in matrix_keys:
        time_matrix.append(travel_times[k])
    time_matrix = np.array(time_matrix, dtype=np.uint16)
    time_matrix = time_matrix.reshape(num_customers, num_customers)
    return time_matrix.tolist()


def generate_twcvrp_dataset(
    num_customers: int,
    num_cities: Optional[int] = None,
    num_depots: int = 1,
    precision=np.uint16,
    is_dynamic: bool = False,
) -> Dict:
    num_cities = num_cities if num_cities else max(1, num_customers // 50)
    dataset = {
        "num_vehicles": [],
        "locations": [],
        "demands": [],
        "time_matrix": [],
        "time_windows": [],
        "appear_times": [],
        "vehicle_capacities": [],
        "map_size": MAP_SIZE,
        "num_cities": num_cities,
        "num_depots": num_depots,
    }
    for _ in tqdm(
        range(NUM_INSTANCES), desc=f"Generating {num_customers} customer instances"
    ):
        instance = generate_twcvrp_instance(
            num_customers, num_cities, num_depots, is_dynamic=is_dynamic
        )
        dataset["locations"].append(instance["locations"].astype(precision))
        dataset["demands"].append(instance["demands"].astype(precision))
        dataset["vehicle_capacities"].append(instance["vehicle_capacity"])
        instance["travel_times"] = {
            k: round(v, 2) for k, v in instance["travel_times"].items()
        }
        time_matrix = get_time_matrix(
            num_customers + num_depots, instance["travel_times"]
        )
        time_matrix = np.array(time_matrix, dtype=np.uint16)
        dataset["time_matrix"].append(time_matrix)
        dataset["time_windows"].append(instance["time_windows"].astype(precision))
        dataset["appear_times"].append(instance["appear_time"])
        dataset["num_vehicles"].append(1)

    return {k: np.array(v) for k, v in dataset.items()}


def main():
    # customer_counts = [10, 20, 50, 100, 200, 500, 1000]
    customer_counts = [10]
    os.makedirs("data/real_twcvrp", exist_ok=True)
    for num_customers in tqdm(customer_counts):
        dataset = generate_twcvrp_dataset(num_customers)
        save_dataset(dataset, f"data/real_twcvrp/twvrp_{num_customers}.npz")


if __name__ == "__main__":
    main()
    dataset = load_dataset("data/real_twcvrp/twvrp_10.npz")
    visualize_instance(dataset)
