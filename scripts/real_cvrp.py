import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from city import Map, map_drawer, Location
from constants import DEPOT


# TODO: put map_size, vehicle_capacity, demand_range in settings file
def generate_cvrp_instance(
    num_customers: int,
    map_size: Tuple[int, int] = (100, 100),
    num_cities: int = 3,
    num_depots: int = 1,
    vehicle_capacity: int = 100,
    demand_range: Tuple[int, int] = (1, 10),
) -> dict:
    map_instance = Map(map_size, num_cities, num_depots)
    locations = map_instance.sample_locations(num_customers)
    depot = next(loc for loc in locations if loc.type == DEPOT)
    locations.remove(depot)
    locations.insert(0, depot)
    demands = np.random.randint(
        demand_range[0], demand_range[1] + 1, size=num_customers
    )
    demands[0] = 0  # Depot has no demand

    return {
        "locations": np.array([(loc.x, loc.y) for loc in locations]),
        "demands": demands,
        "vehicle_capacity": vehicle_capacity,
    }


def generate_cvrp_dataset(
    num_customers: int,
    num_instances: int = 1000,
    map_size: Tuple[int, int] = (100, 100),
    num_cities: int = 3,
    num_depots: int = 1,
    vehicle_capacity: int = 100,
    demand_range: Tuple[int, int] = (1, 10),
) -> dict:
    dataset = {
        "locations": [],
        "demands": [],
        "vehicle_capacities": [],
    }

    for _ in tqdm(
        range(num_instances), desc=f"Generating {num_customers} customer instances"
    ):
        instance = generate_cvrp_instance(
            num_customers,
            map_size,
            num_cities,
            num_depots,
            vehicle_capacity,
            demand_range,
        )
        dataset["locations"].append(instance["locations"])
        dataset["demands"].append(instance["demands"])
        dataset["vehicle_capacities"].append(instance["vehicle_capacity"])

    return {k: np.array(v) for k, v in dataset.items()}


def save_dataset(dataset: dict, filename: str):
    np.savez_compressed(filename, **dataset)


def main():
    customer_counts = [10, 20, 50, 100, 200, 500, 1000]

    for num_customers in tqdm(customer_counts):
        dataset = generate_cvrp_dataset(num_customers)
        save_dataset(dataset, f"data/real_cvrp/cvrp_{num_customers}.npz")


if __name__ == "__main__":
    # main()
    gen_path = "data/real_cvrp/cvrp_10.npz"
    data = np.load(gen_path)
    locations = data["locations"]
    locations = locations[1]
    locations = [Location(loc[0], loc[1]) for loc in locations]
    map = Map((100, 100), 1, 1)
    map.locations = locations
    img = map_drawer(map)
    img.show()
