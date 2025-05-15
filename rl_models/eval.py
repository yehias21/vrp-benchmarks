import os
import glob
import argparse
import json
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from tensordict import TensorDict
from rl4co.models import REINFORCE, POMO


def dict_to_tensordict_cvrp(data: dict, map_size: float = 1000.0) -> TensorDict:
    num_depots = int(data.get('num_depots', 1))
    capacity = torch.tensor(
        data['vehicle_capacities'].astype(np.float32), dtype=torch.float32
    )
    locs_full = torch.tensor(
        data['locations'].astype(np.float32) / map_size, dtype=torch.float32
    )
    demands = torch.tensor(
        data['demands'].astype(np.float32)[:, num_depots:]
        / data['vehicle_capacities'].astype(np.float32),
        dtype=torch.float32,
    )

    depot_pos = locs_full[:, :num_depots].reshape(-1, 2)
    city_pos = locs_full[:, num_depots:]

    return TensorDict(
        {
            'capacity': capacity,
            'demand': demands,
            'depot': depot_pos,
            'locs': city_pos,
        },
        batch_size=torch.Size([capacity.shape[0]]),
    )


def dict_to_tensordict_tw(data: dict, map_size: float = 1000.0,
                           max_time: float = 1440.0) -> TensorDict:
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

    locs_np = stack_field('locations', np.float32, 3)
    batch_sz, num_nodes, _ = locs_np.shape
    locs_norm = torch.tensor(locs_np / map_size, dtype=torch.float32)
    depot_pos = locs_norm[:, :1].view(batch_sz, 1, 2)
    city_pos = locs_norm[:, 1:]

    dem_np = stack_field('demands', np.float32, 2)
    vc_raw = data['vehicle_capacities']
    if isinstance(vc_raw, np.ndarray) and vc_raw.dtype == object:
        vc_list = [float(v[0]) for v in vc_raw.flat]
        cap_np = np.array(vc_list, dtype=np.float32)
    else:
        cap_np = vc_raw.astype(np.float32)

    dem_line_np = dem_np[:, 1:]
    dem_line_norm = dem_line_np / cap_np[:, None]
    dem_linehaul = torch.tensor(dem_line_norm, dtype=torch.float32)
    dem_backhaul = torch.zeros_like(dem_linehaul)

    tw_np = stack_field('time_windows', np.float32, 3)
    tw_tensor = torch.tensor(tw_np, dtype=torch.float32)

    service_time = torch.zeros((batch_sz, num_nodes), dtype=torch.float32)
    speed = torch.ones((batch_sz, num_nodes, num_nodes), dtype=torch.float32)
    distance_limit = torch.full((batch_sz,), float('inf'), dtype=torch.float32)
    open_route = torch.zeros((batch_sz,), dtype=torch.bool)
    vehicle_capacity = torch.tensor(cap_np, dtype=torch.float32)
    capacity_original = vehicle_capacity.clone()

    return TensorDict(
        {
            'locs': city_pos,
            'demand_linehaul': dem_linehaul,
            'demand_backhaul': dem_backhaul,
            'distance_limit': distance_limit,
            'time_windows': tw_tensor,
            'service_time': service_time,
            'vehicle_capacity': vehicle_capacity,
            'capacity_original': capacity_original,
            'open_route': open_route,
            'speed': speed,
        },
        batch_size=torch.Size([batch_sz]),
    )


def evaluate(problem: str, data_dir: str, ckpt_root: str, output_root: str,
             solver_list: list, batch_size: int, map_size: float,
             max_time: float):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if problem == 'cvrp':
        pattern = os.path.join(data_dir, 'cvrp_*_single_depot_single_vehicule_sumDemands.npz')
    else:
        pattern = os.path.join(data_dir, 'twvrp_*_single_depot.npz')

    for npz_path in sorted(glob.glob(pattern)):
        data = np.load(npz_path, allow_pickle=True)
        N = int(os.path.basename(npz_path).split('_')[1])
        problem_name = f"{problem}_{N}"

        for solver in solver_list:
            if problem == 'cvrp':
                model_cls = REINFORCE if solver == 'attention' else POMO
                size_limit = 100
            else:
                model_cls = REINFORCE
                size_limit = 50

            model_size = N if N <= size_limit else size_limit
            ckpt_dir = os.path.join(ckpt_root, f"{solver}_{model_size}")
            preferred = os.path.join(ckpt_dir, 'last-v1.ckpt')
            fallback = os.path.join(ckpt_dir, 'last.ckpt')
            ckpt_path = preferred if os.path.isfile(preferred) else fallback
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")

            model = model_cls.load_from_checkpoint(ckpt_path)
            policy, env = model.policy.to(device), model.env

            # build tensordict
            if problem == 'cvrp':
                td = dict_to_tensordict_cvrp(dict(data), map_size)
            else:
                td = dict_to_tensordict_tw(dict(data), map_size, max_time)

            td_init = env.reset(td=td.clone(), batch_size=[batch_size]).to(device)
            out = policy(td_init.clone(), env, phase='test', decode_type='greedy')

            actions = out['actions'].cpu()
            rewards = out['reward'].cpu()

            # extract locs/depot/tw depending on problem
            if problem == 'cvrp':
                locs_norm, depot_norm = td['locs'].cpu(), td['depot'].cpu()
            else:
                locs_norm, depot_norm = td['locs'].cpu(), td['vehicle_capacity'].cpu()  # placeholder

            out_dir = os.path.join(output_root, solver, problem_name)
            os.makedirs(out_dir, exist_ok=True)

            for i, (r, act_seq) in enumerate(zip(rewards, actions)):
                start = time.time()
                # TODO: rescale coords/time windows and compute metrics per problem
                # save JSON and render as in individual scripts
                print(f"[{problem_name}][{solver}][inst {i}] reward={r:.2f}")

            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate VRP models')
    parser.add_argument('--problem', choices=['cvrp', 'twcvrp'], required=True,
                        help='Which problem to evaluate: cvrp or twcvrp')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing NPZ instances')
    parser.add_argument('--ckpt_root', type=str, required=True,
                        help='Root directory of checkpoints')
    parser.add_argument('--output_root', type=str, default='results',
                        help='Directory to save outputs')
    parser.add_argument('--solver', nargs='+', default=['attention'],
                        help='Solvers to run (attention, pomo)')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for evaluation')
    parser.add_argument('--map_size', type=float, default=1000.0,
                        help='Scaling factor for coordinates')
    parser.add_argument('--max_time', type=float, default=1440.0,
                        help='Max time for TW')
    args = parser.parse_args()
    evaluate(
        problem=args.problem,
        data_dir=args.data_dir,
        ckpt_root=args.ckpt_root,
        output_root=args.output_root,
        solver_list=args.solver,
        batch_size=args.batch_size,
        map_size=args.map_size,
        max_time=args.max_time,
    )
