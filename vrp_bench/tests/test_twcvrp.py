import pytest
import sys, os

# Ensure that the parent directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..')))
from vrp_bench.real_twcvrp import generate_twcvrp_dataset
from vrp_bench.constants import NUM_INSTANCES

@pytest.mark.parametrize(
    "num_vehicles,num_customers",
    [
        (1, 10),
        (1, 20),
        (2, 10),
        (2, 20),
    ],
)
def test_dataset_format(num_vehicles, num_customers):
    dataset = generate_twcvrp_dataset(num_customers, num_vehicles=num_vehicles)
    num_instances = len(dataset["locations"])
    num_depots = 1
    to_check = [
        "locations",
        "time_matrix",
        "demands",
        "time_windows",
        "num_vehicles",
        "num_depots",
        "vehicle_capacities",
        "appear_times",
    ]
    for c in to_check:
        assert c in dataset.keys(), f"{c} not found in dataset keys"

    assert dataset["locations"].shape == (NUM_INSTANCES, num_customers + num_depots, 2)
    assert dataset["time_matrix"].shape == (
        NUM_INSTANCES,
        num_customers + num_depots,
        num_customers + num_depots,
    )
    assert dataset["demands"].shape == (NUM_INSTANCES, num_customers + num_depots)
    assert dataset["time_windows"].shape == (
        NUM_INSTANCES,
        num_customers + num_depots,
        2,
    )
    assert dataset["num_vehicles"].shape == (NUM_INSTANCES,)
    assert dataset["num_depots"] == num_depots
    assert dataset["vehicle_capacities"].shape == (NUM_INSTANCES, num_vehicles)
    assert dataset["appear_times"].shape == (NUM_INSTANCES, num_customers + num_depots)
