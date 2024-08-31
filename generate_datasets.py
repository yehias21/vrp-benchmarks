import numpy as np
from rl4co.data.generate_data import generate_vrp_data


vrp_sizes = [10, 20, 50, 100, 200, 500, 1000]
dataset_size = 1000
datasets = []
for size in vrp_sizes:
    datasets.append(generate_vrp_data(dataset_size, size))


np.save("datasets.npy", datasets)
