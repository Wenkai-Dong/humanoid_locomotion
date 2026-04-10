import argparse
from pathlib import Path
import h5py

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to filter an HDF5 dataset and extract specific episodes based on a target count.")
parser.add_argument("input", type=str, help="The path to the input HDF5 file.")
# parse the arguments
args_cli = parser.parse_args()

source_path = args_cli.input
p = Path(source_path)
target_path= p.with_name("filter_" + p.name)

target_value = 1
dataset_name = 'episode_count'

with h5py.File(source_path, 'r') as f_old, h5py.File(target_path, 'w') as f_new:
    old_data_group = f_old['data']
    new_data_group = f_new.create_group('data')
    new_index = 0

    all_keys = sorted(list(old_data_group.keys()), key=lambda x: int(x.split('_')[-1]))

    for key in all_keys:
        demo_group = old_data_group[key]
        count_array = demo_group[dataset_name][:]
        if count_array[-1] == target_value:
            new_name = f'demo_{new_index}'
            f_old.copy(demo_group, new_data_group, name=new_name)
            new_index += 1