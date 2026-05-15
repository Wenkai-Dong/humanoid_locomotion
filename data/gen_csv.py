import re
import h5py
import pandas as pd
import numpy as np
from pathlib import Path

hdf5_dir = Path("C:\\Users\\395\\OneDrive\\data\\Attention-G1-v3\\mha\\2026-05-10_02-48-10\\model_21000")
criteria_records = []
error_recorders = []

for hdf5_path in sorted(hdf5_dir.glob("*.hdf5")):
    parts = hdf5_path.parts
    cmd_vel = float(re.search(r"lin_vel_x_(-?\d+\.?\d*)", hdf5_path.stem).group(1))

    with h5py.File(hdf5_path, 'r') as f:
        data_group = f["data"]
        for demo_name in data_group:
            demo = f['data'][demo_name]
            if f['data'][demo_name]["episode_count"][...][0] != 1:
                continue

            linear_tracking_error = demo["linear_tracking_error"][...]
            angular_tracking_error = demo["angular_tracking_error"][...]
            demo_id = int(demo_name.split("_")[1])

            for err_type, arr in [("linear", linear_tracking_error), ("angular", angular_tracking_error)]:
                row = {
                    "env": parts[-5],
                    "agent": parts[-4],
                    "time": parts[-3],
                    "checkpoint": parts[-2],
                    "cmd_vel": cmd_vel,
                    "demo_id": demo_id,
                    "type": err_type,
                }
                for step, val in enumerate(arr):
                    row[step] = val[0]
                error_recorders.append(row)

            current_criteria_records = {
                "env": parts[-5],
                "agent": parts[-4],
                "time": parts[-3],
                "checkpoint": parts[-2],
                "cmd_vel": cmd_vel,
                "demo_id": int(demo_name.split("_")[1]),
                "success": demo.attrs["success"],
                "num_samples": len(linear_tracking_error),
                "termination": demo["Termination"][...][0],
                "sub_terrain": demo["sub_terrain"][...][0],
                "mean_linear": np.mean(linear_tracking_error),
                "mean_angular": np.mean(angular_tracking_error),
            }
            criteria_records.append(current_criteria_records)

df_error = pd.DataFrame(error_recorders)
df_criteria = pd.DataFrame(criteria_records)

df_error = df_error.sort_values(by=["cmd_vel", "demo_id", "type"]).reset_index(drop=True)
df_criteria = df_criteria.sort_values(by=["cmd_vel", "demo_id"]).reset_index(drop=True)

out_dir = hdf5_dir
df_error.to_csv(out_dir / "tracking_error.csv", index=False)
df_criteria.to_csv(out_dir / "criteria.csv", index=False)