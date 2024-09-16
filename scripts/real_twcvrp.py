import numpy as np
from typing import List, Tuple
from city import Map, Location, map_drawer
import random
from tqdm import tqdm
from travel_time_generator import sample_travel_time, get_distances


def generate_time_window(earliest_time: int, latest_time: int) -> Tuple[int, int]:
    start = random.randint(
        earliest_time, latest_time - 60
    )  # At least 60 minutes window
    end = random.randint(start + 60, latest_time)
    return start, end


def generate_dataset(
    num_customers: int,
    num_instances: int = 1000,
    num_depots: int = 1,
    map_size: Tuple[int, int] = (100, 100),
    demand_range: Tuple[int, int] = (1, 10),
) -> List[dict]:
    datasets = []
    for _ in range(num_instances):
        map_instance = Map(
            map_size, num_cities=max(1, num_customers // 50), num_depots=num_depots
        )
        locations = map_instance.sample_locations(num_customers + num_depots)

        distances = get_distances(map_instance)

        travel_times = {}
        for i in range(len(locations)):
            for j in range(len(locations)):
                if i != j:
                    current_time = random.randint(
                        0, 1440
                    )  # Random start time for each trip
                    travel_times[(i, j)] = sample_travel_time(
                        i, j, distances, current_time
                    )
                else:
                    travel_times[(i, j)] = 0

        time_windows = [
            generate_time_window(0, 1440) for _ in range(num_customers + num_depots)
        ]

        demands = np.random.randint(
            demand_range[0], demand_range[1], size=num_customers + num_depots
        )
        demands[:num_depots] = 0  # Depots have no demand

        dataset = {
            "locations": np.array([(loc.x, loc.y) for loc in locations]),
            "travel_times": travel_times,
            "time_windows": np.array(time_windows),
            "demands": demands,
            "num_depots": num_depots,
        }
        datasets.append(dataset)

    return datasets


def save_datasets(datasets: List[dict], filename: str):
    np.savez_compressed(filename, datasets=datasets)


def main():
    customer_counts = [10, 20, 50, 100, 200, 500, 1000]

    for num_customers in tqdm(customer_counts):
        datasets = generate_dataset(num_customers)
        filename = f"data/real_twcvrp/twvrp_{num_customers}.npz"
        save_datasets(datasets, filename)


if __name__ == "__main__":
    # main()
    gen_path = "data/real_twcvrp/twvrp_10.npz"
    data = np.load(gen_path, allow_pickle=True)
    print(data["datasets"][0].keys())
    data = data["datasets"][0]
    locations = data["locations"]
    locations = [Location(loc[0], loc[1]) for loc in locations]
    map = Map((100, 100), 1, 1)
    map.locations = locations
    img = map_drawer(map)
    img.show()
