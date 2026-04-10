import h5py
import pandas as pd
import numpy as np

hdf5_path = 'D:/humanoid_locomotion/logs/rsl_rl/ame1_stage1_h1_v0/2026-03-25_19-24-10/dataset/model_19999_1.0.hdf5'

records = []
with h5py.File(hdf5_path, 'r') as f:
    for demo_name in f['data']:
        demo = f['data'][demo_name]
        records.append({
            'demo': demo_name,
            'Termination': int(np.array(demo['Termination'][()]).flat[0]),
            'sub_terrain': int(np.array(demo['sub_terrain'][()]).flat[0]),
            'survival_steps': demo['angular_tracking_error'].shape[0],
        })

df = pd.DataFrame(records)

# ============ sub_terrain == 3 ============
sub3 = df[df['sub_terrain'] == 3]

print(f"sub_terrain == 3 总数: {len(sub3)}")
print()

print("=== Termination 分布 ===")
term_map = {0: 'time_out', 1: 'base_height', 2: 'bad_orientation', 3: 'success'}
counts = sub3['Termination'].value_counts().sort_index()
for t, c in counts.items():
    name = term_map.get(t, f'unknown({t})')
    print(f"  {t} ({name}): {c} 个, 占比 {c/len(sub3)*100:.1f}%")
print()

print("=== 存活步数 (survival_steps) ===")
print(f"  均值: {sub3['survival_steps'].mean():.1f}")
print(f"  中位数: {sub3['survival_steps'].median():.1f}")
print(f"  最小: {sub3['survival_steps'].min()}")
print(f"  最大: {sub3['survival_steps'].max()}")
print(f"  标准差: {sub3['survival_steps'].std():.1f}")
print()

print("=== 各 Termination 的存活步数 ===")
for t, group in sub3.groupby('Termination'):
    name = term_map.get(t, f'unknown({t})')
    print(f"  {t} ({name}): 均值={group['survival_steps'].mean():.1f}, "
          f"中位数={group['survival_steps'].median():.1f}, "
          f"范围=[{group['survival_steps'].min()}, {group['survival_steps'].max()}]")