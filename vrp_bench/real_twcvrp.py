import os
from typing import Dict, Tuple, Optional
import random

import numpy as np
from numpy import dtype
from tqdm import tqdm

from travel_time_generator import sample_travel_time, get_distances
from common import (
    generate_base_instance,
    save_dataset,
    load_dataset,
    visualize_instance,
)
from time_windows_generator import sample_time_window
from constants import NUM_INSTANCES, DEMAND_RANGE, MAP_SIZE, REALIZATIONS_PER_MAP


def generate_time_window(customer_appear_time: int) -> Tuple[int, int]:
    return sample_time_window(random.randint(0, 1), customer_appear_time)


def generate_twcvrp_instance(
    num_customers: int,
    num_cities: Optional[int] = None,
    instance: Optional[Dict] = None,
    num_depots: int = 1,
    is_dynamic: bool = False,
) -> Dict:
    num_cities = num_cities if num_cities else max(1, num_customers // 50)
    if not instance:
        instance = generate_base_instance(
            num_customers,
            MAP_SIZE,
            num_cities,
            num_depots,
            DEMAND_RANGE,
            is_dynamic,
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


def get_num_vehicles(time_windows):
    tw = np.copy(np.array(list(time_windows))).flatten()
    from matplotlib import pyplot as plt
    return np.max(plt.hist(tw, bins=24, range=(0,1440))[0]).astype(dtype=np.uint16)


def generate_twcvrp_dataset(
    num_customers: int,
    num_cities: Optional[int] = None,
    num_depots: int = 1,
    num_vehicles: int = 1,
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
        "travel_times": [],
        "map_size": MAP_SIZE,
        "num_cities": num_cities,
        "num_depots": num_depots,
    }
    for index in tqdm(
        range(NUM_INSTANCES), desc=f"Generating {num_customers} customer instances"
    ):

        if index % REALIZATIONS_PER_MAP == 0:
            instance = generate_twcvrp_instance(
                num_customers=num_customers,
                num_cities=num_cities,
                instance=None,
                num_depots=num_depots,
                is_dynamic=is_dynamic,
            )
        else:
            instance = generate_twcvrp_instance(
                num_customers=num_customers,
                num_cities=num_cities,
                instance=instance,
                num_depots=num_depots,
                is_dynamic=is_dynamic,
            )
        # dynamic num_vehicles
        num_vehicles = get_num_vehicles(instance["time_windows"])
        #
        dataset["locations"].append(instance["locations"].astype(precision))
        dataset["demands"].append(instance["demands"].astype(precision))
        dataset["vehicle_capacities"].append(
            [instance["vehicle_capacity"]] * num_vehicles
        )
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
        dataset["num_vehicles"].append(num_vehicles)
        dataset["travel_times"].append(instance['travel_times'])

    return {k: np.array(v) for k, v in dataset.items()}


def main():
    customer_counts = [10, 20, 50, 100, 200, 500, 1000]
    os.makedirs("../data/real_twcvrp", exist_ok=True)
    for num_customers in tqdm(customer_counts):
        depots = max(1, num_customers // 100)
        dataset = generate_twcvrp_dataset(num_customers, num_depots=depots)
        save_dataset(dataset, f"../data/real_twcvrp/twvrp_{num_customers}.npz")


if __name__ == "__main__":
    main()
    dataset = load_dataset("../data/real_twcvrp/twvrp_1000.npz")
    visualize_instance(dataset)
