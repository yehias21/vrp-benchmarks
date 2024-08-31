import numpy as np
from rl4co.data.generate_data import generate_vrp_data


vrp_sizes = [10, 20, 50, 100, 200, 500, 1000]
dataset_size = 1000
datasets = []
for vrp_size in vrp_sizes:
    data = generate_vrp_data(dataset_size, vrp_size)
    data_path = "data/cvrp/vrp_{}_{}.npz".format(vrp_size, dataset_size)
    np.savez(data_path, **data)
