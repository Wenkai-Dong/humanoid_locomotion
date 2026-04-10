import argparse
import pandas as pd
import os
# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to filter an HDF5 dataset and extract specific episodes based on a target count.")
parser.add_argument("input", type=str, help="The path to the input HDF5 file.")
# parse the arguments
args_cli = parser.parse_args()

import h5py
import numpy as np

file_path = args_cli.input
terrain = ["Rough", "Stairs", "StairsInverted", "Gaps", "GridStones", "Pallets", "Pits", "PitsInverted", "Beams"]
# termination = ["time_out", "base_height", "bad_orientation", "base_contact", "success"]
key_terrain = "sub_terrain"
key_lin_err = "linear_tracking_error"
key_ang_err = "angular_tracking_error"

stats = {}
for i in range(9):
    stats[i] = {
        "lin_total": 0.0, "ang_total": 0.0, "total_steps": 0,
        "demo_count": 0, "success_count": 0,
        "termination_count": {"0": 0, "1": 0, "2": 0, "3": 0}
    }

with h5py.File(file_path, 'r') as f:
    data_group = f['data']
    for key in data_group:
        demo = data_group[key]

        terrain_id = int(demo['sub_terrain'][0])
        stats[terrain_id]['demo_count'] += 1

        is_success = demo.attrs['success']
        if is_success:
            stats[terrain_id]['success_count'] += 1
            lin_data = demo['linear_tracking_error'][:]
            ang_data = demo['angular_tracking_error'][:]
            stats[terrain_id]['lin_total'] += np.sum(lin_data)
            stats[terrain_id]['ang_total'] += np.sum(ang_data)
            stats[terrain_id]['total_steps'] += len(lin_data)
        else:
            termination_id = demo["Termination"][-1]
            stats[terrain_id]['termination_count'][f"{termination_id}"] += 1


w_name = 16
w_lin  = 21
w_ang  = 22
w_succ = 13
w_fail = 15
header = (
    f"{'Terrain':<{w_name}} | "
    f"{'linear tracking error':<{w_lin}} | "
    f"{'angular tracking error':<{w_ang}} | "
    f"{'success rate':<{w_succ}} | "
    f"{'time out':<{w_fail}} | "
    f"{'base height':<{w_fail}} | "
    f"{'bad orientation':<{w_fail}} | "
    f"{'base contact':<{w_fail}}"
)
print(header)
for i in range(9):
    bucket = stats[i]
    steps = bucket['total_steps']
    if steps == 0:
        avg_lin = -1
        avg_ang = -1
    else:
        avg_lin = bucket['lin_total'] / steps
        avg_ang = bucket['ang_total'] / steps

    succ_rate = (bucket['success_count'] / bucket['demo_count']) * 100.0
    timeout = (bucket["termination_count"]["0"]/bucket["demo_count"])*100.0
    base_height = (bucket["termination_count"]["1"]/bucket["demo_count"])*100.0
    bad_orientation = (bucket["termination_count"]["2"]/bucket["demo_count"])*100.0
    base_contact = (bucket["termination_count"]["3"]/bucket["demo_count"])*100.0

    row_str = (
        f"  {terrain[i]:<{w_name - 2}} | "
        f"  {avg_lin:<{w_lin - 2}.5f} | "  # 留2格padding，看起来更舒服
        f"  {avg_ang:<{w_ang - 2}.5f} | "
        f"  {succ_rate:<{w_succ - 2}.2f} | "  # % 符号也占位
        f"  {timeout:<{w_fail - 2}.2f} | "
        f"  {base_height:<{w_fail - 2}.2f} | "
        f"  {bad_orientation:<{w_fail - 2}.2f} | "
        f"  {base_contact:<{w_fail - 2}.2f}"  # 最后一列不需要 |
    )
    print(row_str)