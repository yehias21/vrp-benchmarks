import numpy as np
from typing import List, Tuple, Dict
from city import Map, Location
from constants import DEPOT, DYNAMIC_PERCENTAGE
import random


def generate_base_instance(
    num_customers: int,
    map_size: Tuple[int, int],
    num_cities: int,
    num_depots: int,
    demand_range: Tuple[int, int],
    is_dynamic: bool = False,
) -> Dict:
    map_instance = Map(map_size, num_cities, num_depots)
    locations = map_instance.sample_locations(num_customers)

    demands = np.random.randint(
        demand_range[0], demand_range[1] + 1, size=num_customers + num_depots
    )
    demands[0] = 0  # Depot has no demand
    appear_time = []
    if is_dynamic:
        num_dynamic_customers = int(num_customers * DYNAMIC_PERCENTAGE)
        dynamic_customers = random.sample(range(num_customers), num_dynamic_customers)

        for i in range(num_customers + num_depots):
            if i in dynamic_customers:
                appear_time.append(random.uniform(0, 1440))
            else:
                appear_time.append(0)
    else:
        appear_time = [0] * (num_customers + num_depots)

    return {
        "locations": np.array([(loc.x, loc.y) for loc in locations]),
        "demands": demands,
        "map_instance": map_instance,
        "vehicle_capacity": int(
            (random.random() * 0.5 + 0.5) * max(demands) * num_customers
        ),
        "appear_time": np.array(appear_time),
    }


def save_dataset(dataset: Dict, filename: str):
    np.savez_compressed(filename, **dataset)


def load_dataset(filename: str) -> Dict:
    return dict(np.load(filename, allow_pickle=True))


def visualize_instance(dataset: Dict, index: int = 0):
    from city import map_drawer

    locations = dataset["locations"][index]
    map_instance = Map(
        dataset["map_size"], dataset["num_cities"], dataset["num_depots"]
    )
    map_instance.locations = [Location(loc[0], loc[1]) for loc in locations]
    img = map_drawer(map_instance)
    img.show()
