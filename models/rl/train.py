#!/usr/bin/env python
import argparse
import random
import math
import numpy as np
from rl4co.envs.routing import CVRPEnv
from rl4co.envs.routing.mtvrp.generator import MTVRPGenerator
from rl4co.envs.routing.mtvrp.env import MTVRPEnv
from rl4co.models import AttentionModelPolicy, REINFORCE, POMO
from rl4co.utils.trainer import RL4COTrainer
from lightning.pytorch.callbacks import ModelCheckpoint
import torch

# --- Constants for TW variant ---
DEMAND_RANGE = (1, 10)
MAP_SIZE = (1000, 1000)
DYNAMIC_PERCENTAGE = 0.3

def normal_distribution(x, mean, std_dev):
    return math.exp(-((x - mean) ** 2) / (2 * std_dev ** 2)) / (std_dev * math.sqrt(2 * math.pi))

def time_factor(current_time):
    return 0.5 + 2 * (normal_distribution(current_time, 480, 90)
                      + normal_distribution(current_time, 1020, 90))

def random_factor(current_time):
    ebb = normal_distribution(current_time, 480, 90) + normal_distribution(current_time, 1020, 90)
    mu = 0 + 0.1 * ebb
    sigma = 0.3 + 0.2 * ebb
    return random.lognormvariate(mu, sigma)

def sample_accidents(current_time):
    rate = 0.05 * normal_distribution(current_time, 1260, 120)
    rate = max(rate, 0)
    return np.random.poisson(lam=rate)

def calculate_delay(distance, current_time):
    tf = time_factor(current_time)
    df = 1 - math.exp(-distance / 50)
    base = 0.25 * tf * df
    rnd = random_factor(current_time)
    delay = base * rnd
    accidents = sample_accidents(current_time)
    if accidents > 0:
        arr = np.random.uniform(30, 120, size=accidents)
        delay += float(np.sum(arr))
    return delay

def get_distances(map_inst):
    dist = {}
    L = map_inst.locations
    for i in range(len(L)):
        for j in range(len(L)):
            dist[(i, j)] = L[i].distance(L[j])
    return dist

def sample_travel_time(i, j, distances, current_time, velocity=1):
    return distances[(i, j)] / velocity + calculate_delay(distances[(i, j)], current_time)

def sample_time_window(mode, appear_time):
    tw_starts = [appear_time, appear_time + 60]
    tw_ends = [appear_time + 180, appear_time + 240]
    return (tw_starts[mode], tw_ends[mode])

def appear_time_sampler(appear_times_array, idx):
    return appear_times_array[idx]

def time_window_sampler(appear_time, idx, rnd):
    return sample_time_window(rnd, appear_time)

def demand_sampler(demands, idx):
    return int(demands[idx])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL4CO on VRP variants (CVRP or VRPTW)")
    parser.add_argument("--num_loc", type=int, required=True,
                        help="Number of customer locations")
    parser.add_argument("--algo", type=str, choices=["attention", "pomo"], required=True,
                        help="Which RL algorithm to use")
    parser.add_argument("--batch_size", type=int, required=True,
                        help="Batch size for rl")
    parser.add_argument("--variant", type=str, choices=["cvrp", "twvrp"], required=True,
                        help="Problem variant to train: 'cvrp' or 'twvrp'")
    args = parser.parse_args()

    # Environment setup
    if args.variant == "cvrp":
        env = CVRPEnv(generator_params={'num_loc': args.num_loc})
    else:
        generator = MTVRPGenerator(
            num_loc=args.num_loc,
            variant_preset="vrptw",
            max_time=1440,
            map_size=MAP_SIZE,
            num_cities=max(1, 100 // 50),
            num_depots=1,
            demand_sampler=lambda inst, i: demand_sampler(inst["demands"], i),
            appear_time_sampler=lambda inst, i: appear_time_sampler(inst["appear_time"], i),
            travel_time_sampler=lambda inst, i, j: sample_travel_time(
                i, j,
                get_distances(inst["map_instance"]),
                random.randint(0, 1440)
            ),
            time_window_sampler=lambda inst, i: time_window_sampler(
                inst["appear_time"][i],
                i,
                random.randint(0, 1)
            ),
        )
        env = MTVRPEnv(generator)

    # Policy backbone
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=128,
        num_encoder_layers=3,
        num_heads=8,
    )

    # Model selection
    if args.algo == "attention":
        model_cls = REINFORCE
        baseline = "rollout"
    else:
        model_cls = POMO
        baseline = "shared"

    # Training sizes per variant
    if args.variant == "cvrp":
        train_size = 100_000
        val_size = 1_000
    else:
        train_size = 1_000_000
        val_size = 1_000

    model = model_cls(
        env,
        policy,
        baseline=baseline,
        batch_size=args.batch_size,
        train_data_size=train_size,
        val_data_size=val_size,
        optimizer_kwargs={"lr": 1e-4},
    )

    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{args.variant}/{args.algo}_{args.num_loc}",
        filename="epoch_{epoch:03d}",
        save_top_k=1,
        save_last=True,
        monitor="val/reward",
        mode="max",
    )

    # Trainer
    trainer = RL4COTrainer(
        max_epochs=10,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
    )

    print(f"[START] variant={args.variant} algo={args.algo} num_loc={args.num_loc} batch_size={args.batch_size}")
    trainer.fit(model)
