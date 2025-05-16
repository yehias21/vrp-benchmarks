import os
import glob
import argparse
import json
import time
import gc

import numpy as np
import torch
import matplotlib.pyplot as plt
from tensordict import TensorDict
from rl4co.models import REINFORCE, POMO

# ----------------
# Helper Functions
# ----------------

def dict_to_tensordict_cvrp(data: dict, map_size=1000) -> TensorDict:
    """Convert CVRP data to TensorDict format"""
    num_depots = int(data.get('num_depots', 1))

    capacity = torch.tensor(data['vehicle_capacities'].astype(np.float32), dtype=torch.float32)
    locs_full = torch.tensor(data['locations'].astype(np.float32)/map_size, dtype=torch.float32)
    demands = torch.tensor(data['demands'].astype(np.float32)[:,num_depots:]\
                          /data['vehicle_capacities'].astype(np.float32), dtype=torch.float32)

    depot_pos = locs_full[:,:num_depots].reshape(-1,2)     # shape [num_depots, 2]
    city_pos = locs_full[:, num_depots:]     # shape [num_cities, 2]

    # Build the TensorDict
    td = TensorDict(
        {
            'capacity': capacity,   # [1,1]
            'demand': demands,     # [1, num_cities]
            'depot': depot_pos,      # [1,2] or [1,num_depots,2]
            'locs': city_pos,       # [1, num_cities, 2]
        },
        batch_size=torch.Size([capacity.shape[0]]),
    )
    return td

def dict_to_tensordict_tw(data: dict, map_size=1000.0, max_time=1440.0) -> TensorDict:
    """Convert VRPTW data to TensorDict format"""
    # Helper to stack object arrays
    def stack_field(name, dtype, ndim):
        raw = data[name]
        if isinstance(raw, np.ndarray) and raw.dtype == object:
            arrs = [np.asarray(x, dtype=dtype) for x in raw]
            arr = np.stack(arrs, axis=0)
        else:
            arr = raw.astype(dtype)
        if arr.ndim == ndim - 1:
            arr = arr[None, ...]
        return arr

    # 1) locations
    locs_np = stack_field('locations', np.float32, 3)
    batch_sz, num_nodes, _ = locs_np.shape
    num_depots = 1
    num_cities = num_nodes - num_depots
    locs_norm = torch.tensor(locs_np / map_size, dtype=torch.float32)
    depot_pos = locs_norm[:, :num_depots].view(batch_sz, num_depots, 2)
    city_pos = locs_norm[:, num_depots:]  # [batch, num_cities,2]

    # 2) demands raw
    dem_np = stack_field('demands', np.float32, 2)

    # 3) capacities
    vc_raw = data['vehicle_capacities']
    if isinstance(vc_raw, np.ndarray) and vc_raw.dtype == object:
        vc_list = [float(v[0]) for v in vc_raw.flat]
        cap_np = np.array(vc_list, dtype=np.float32)
    else:
        cap_np = vc_raw.astype(np.float32)

    # 4) normalize linehaul demand by capacity
    dem_line_np = dem_np[:, num_depots:]                       # [batch, num_cities]
    dem_line_norm = dem_line_np / cap_np[:, None]
    dem_linehaul = torch.tensor(dem_line_norm, dtype=torch.float32)

    # 5) backhaul = zeros
    dem_backhaul = torch.zeros_like(dem_linehaul)

    # 6) time_windows
    tw_np = stack_field('time_windows', np.float32, 3)
    tw_tensor = torch.tensor(tw_np, dtype=torch.float32)    # [batch,num_nodes,2]

    # 7) service_time, speed, distance_limit, open_route
    service_time = torch.zeros((batch_sz, num_nodes), dtype=torch.float32)
    speed = torch.ones((batch_sz, num_nodes, num_nodes), dtype=torch.float32)
    distance_limit = torch.full((batch_sz,), float('inf'), dtype=torch.float32)
    open_route = torch.zeros((batch_sz,), dtype=torch.bool)

    # 8) vehicle capacities as tensors
    vehicle_capacity = torch.tensor(cap_np, dtype=torch.float32)       # [batch]
    capacity_original = vehicle_capacity.clone()

    return TensorDict(
        {
            'locs': city_pos,             # [batch, n_cities, 2]
            'demand_linehaul': dem_linehaul,         # [batch, n_cities] *normalized*
            'demand_backhaul': dem_backhaul,         # [batch, n_cities]
            'distance_limit': distance_limit,       # [batch]
            'time_windows': tw_tensor,            # [batch, num_nodes, 2]
            'service_time': service_time,         # [batch, num_nodes]
            'vehicle_capacity': vehicle_capacity,     # [batch]
            'capacity_original': capacity_original,    # [batch]
            'open_route': open_route,           # [batch]
            'speed': speed,                # [batch, num_nodes, num_nodes]
            'depot': depot_pos,            # [batch, num_depots, 2]
        },
        batch_size=torch.Size([batch_sz]),
    )

def save_results(out_dir, problem_name, solver, i, route_idx, all_coords, length, runtime, reward, 
                td_init, act_seq, env, time_windows=None):
    """Save results as JSON and render visualization"""
    # Assemble JSON result
    result = {
        "problem": problem_name,
        "solver": solver,
        "instance": i,
        "routes": [route_idx],
        "locations": [
            [int(x.item()) for x in coord]
            for coord in all_coords
        ],
        "depots": [0],
        "customers": list(range(1, all_coords.shape[0])),
        "metrics": {
            "total_cost": length,
            "runtime": runtime,
            "reward": reward
        }
    }
    
    # Add time windows if present
    if time_windows is not None:
        result["time_windows"] = [[int(t) for t in tw] for tw in time_windows]

    # Write JSON to disk
    json_path = os.path.join(out_dir, f"instance_{i}.json")
    with open(json_path, "w") as jf:
        json.dump(result, jf, indent=2)

    # Render & save plot
    env.render(td_init, act_seq)  # draws to current matplotlib figure
    plt.axis('off')
    img_path = os.path.join(out_dir, f"instance_{i}.png")
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Instance {i} → saved JSON: {json_path}, image: {img_path}")
    print(f"   total_cost={length:.2f}, runtime={runtime:.3f}s, reward={reward:.2f}")

def evaluate_cvrp(data_pattern, output_dir, solvers=("attention", "pomo"), map_size=1000, batch_size=10):
    """Evaluate models on CVRP instances"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for npz_path in sorted(glob.glob(data_pattern)):
        # Load data and set up names
        data = np.load(npz_path, allow_pickle=True)
        N = int(os.path.basename(npz_path).split("_")[1])
        problem_name = f"cvrp_{N}_single_depot_single_vehicule_sumDemands"

        # Run both attention and pomo policies
        for solver in solvers:
            # Choose model size (use larger size for larger problems)
            model_size = N if N <= 100 else 100
            ckpt_dir = f"/l/users/ahmed.heakl/vrp/checkpoints/{solver}_{model_size}"

            # Prefer last-v1.ckpt, otherwise fallback to last.ckpt
            preferred = os.path.join(ckpt_dir, "last-v1.ckpt")
            fallback = os.path.join(ckpt_dir, "last.ckpt")
            if os.path.isfile(preferred):
                ckpt_path = preferred
            elif os.path.isfile(fallback):
                ckpt_path = fallback
            else:
                raise FileNotFoundError(f"No checkpoint in {ckpt_dir}; looked for last-v1.ckpt and last.ckpt")

            # Load the right model class
            trained_model = (
                REINFORCE.load_from_checkpoint(ckpt_path)
                if solver == "attention"
                else POMO.load_from_checkpoint(ckpt_path)
            )

            print(f"Loaded {solver} model for N={N} from {ckpt_path}")
            
            # Convert data to TensorDict
            td = dict_to_tensordict_cvrp(data, map_size=map_size)
            
            # Initialize environment
            new_policy, env = trained_model.policy, trained_model.env
            td_init = env.reset(td=td.clone(), batch_size=[batch_size]).to(device)
            
            # Ensure output directory exists
            out_dir = os.path.join(output_dir, solver, problem_name)
            os.makedirs(out_dir, exist_ok=True)
            
            # Run policy
            policy = new_policy.to(device)
            out = policy(td_init.clone(), env, phase="test", decode_type="greedy")

            # Extract actions & (normalized) locs/depot
            actions = out["actions"].cpu()    # [batch, seq_len]
            rewards = out["reward"].cpu()     # [batch]
            locs_norm = td["locs"].cpu()      # [batch, n_cities, 2]
            depot_norm = td["depot"].cpu()    # [batch, 2]

            for i, (r, act_seq) in enumerate(zip(rewards, actions)):
                start_time = time.time()
                
                # Scale & round to ints
                depot_i = depot_norm[i].mul(map_size).round().to(torch.int)    # (2,)
                city_coords = locs_norm[i].mul(map_size).round().to(torch.int)     # (n_cities, 2)

                # Build array of all nodes: [depot, city1, city2, ...]
                all_coords = torch.vstack([
                    depot_i.unsqueeze(0),   # index 0 = depot
                    city_coords            # indices 1..N = customers
                ])  # shape: (n_cities+1, 2)

                # Translate greedy actions (which index into customers-only) into full-route
                route_idx = [0] + [int(a) for a in act_seq.tolist()] + [0]

                # Compute true L2 tour length
                route_coords = all_coords[route_idx]
                steps = route_coords[1:] - route_coords[:-1]
                dists = torch.norm(steps.float(), dim=1)
                length = float(dists.sum().item())

                reward = float(r.item())
                runtime = time.time() - start_time

                # Save results and visualize
                save_results(out_dir, problem_name, solver, i, route_idx, all_coords, 
                            length, runtime, reward, td_init[i], act_seq, env)
                
            del trained_model, new_policy, policy, out, td_init
            torch.cuda.empty_cache()
            gc.collect()

def evaluate_vrptw(data_pattern, output_dir, solvers=("attention",), map_size=1000, max_time=1440, batch_size=10):
    """Evaluate models on VRPTW instances"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for npz_path in sorted(glob.glob(data_pattern)):
        # Extract N from filename
        N = int(os.path.basename(npz_path).split("_")[1])
        problem_name = f"twvrp_{N}_single_depot"
        
        # Load data
        data = np.load(npz_path, allow_pickle=True)
        
        for solver in solvers:
            # Pick the model size: use solver_N up to N=50, otherwise solver_50
            model_size = N if N <= 50 else 50
            ckpt_path = os.path.join("/home/yahia.shaaban/twvrp",
                                    f"{solver}_{model_size}",
                                    "last.ckpt")

            # Load model for this problem size
            model = REINFORCE.load_from_checkpoint(ckpt_path)
            policy, env = model.policy, model.env
            policy = policy.to(device)

            print(f"Loaded {solver} model for N={N} from {ckpt_path}")

            # Create output directory
            out_dir = os.path.join(output_dir, solver, problem_name)
            os.makedirs(out_dir, exist_ok=True)

            # Build tensordict
            td = dict_to_tensordict_tw(dict(data), map_size=map_size, max_time=max_time)
            td_init = env.reset(td=td.clone(), batch_size=[batch_size]).to(device)

            # Rollout greedy
            out = policy(td_init.clone(), env, phase="test", decode_type="greedy")
            actions = out["actions"].cpu()   # [batch, seq_len]
            rewards = out["reward"].cpu()    # [batch]
            locs_norm = td["locs"].cpu()     # [batch, num_cities, 2]
            depot_norm = td["depot"].cpu()   # [batch, num_depots, 2]
            tw_norm = td["time_windows"].cpu()

            for i, (r, act_seq) in enumerate(zip(rewards, actions)):
                tic = time.time()

                # Rescale coords
                depot_i = depot_norm[i].mul(map_size).round().to(torch.int).squeeze(0)
                city_coords = locs_norm[i].mul(map_size).round().to(torch.int)

                # Rescale time windows
                city_tw = tw_norm[i].mul(max_time).round().to(torch.int)

                # Assemble node arrays
                all_coords = torch.vstack([depot_i.unsqueeze(0), city_coords])
                all_tw = city_tw

                # Full route indices
                route_idx = [0] + [int(a) for a in act_seq.tolist()] + [0]

                # Compute true tour length
                route_coords = all_coords[route_idx]
                steps = route_coords[1:] - route_coords[:-1]
                length = float(torch.norm(steps.float(), dim=1).sum().item())

                runtime = time.time() - tic
                reward = float(r.item())

                # Save results and visualize
                save_results(out_dir, problem_name, solver, i, route_idx, all_coords, 
                            length, runtime, reward, td_init[i], act_seq, env, all_tw)

            # Cleanup
            del model, policy, env, td, td_init, out
            torch.cuda.empty_cache()
            gc.collect()

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='Evaluate RL models on VRP problems')
    parser.add_argument('--problem_type', type=str, required=True, choices=['cvrp', 'vrptw'], 
                        help='Problem type to evaluate')
    parser.add_argument('--data_pattern', type=str, required=True,
                        help='Glob pattern to find data files')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--solvers', type=str, nargs='+', default=['attention'],
                        help='Solvers to evaluate (e.g., attention, pomo)')
    parser.add_argument('--map_size', type=float, default=1000.0,
                        help='Map size for scaling coordinates')
    parser.add_argument('--max_time', type=float, default=1440.0,
                        help='Maximum time for VRPTW time windows')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for evaluation')

    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate based on problem type
    if args.problem_type == 'cvrp':
        evaluate_cvrp(
            args.data_pattern, 
            args.output_dir, 
            solvers=args.solvers,
            map_size=args.map_size,
            batch_size=args.batch_size
        )
    elif args.problem_type == 'vrptw':
        evaluate_vrptw(
            args.data_pattern, 
            args.output_dir, 
            solvers=args.solvers,
            map_size=args.map_size,
            max_time=args.max_time,
            batch_size=args.batch_size
        )

if __name__ == "__main__":
    main() 