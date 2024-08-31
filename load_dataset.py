import numpy as np

dataset_path = "data/cvrp/vrp_10_1000.npz"
data = np.load(dataset_path)
# NpzFile 'data/cvrp/vrp_10_1000.npz' with keys: depot, locs, demand, capacity
instances = []
for i in range(len(data["locs"])):
    instances.append(
        {
            "depot": data["depot"][i],
            "locs": data["locs"][i],
            "demand": data["demand"][i],
            "capacity": data["capacity"][i],
        }
    )
print(instances[0])
