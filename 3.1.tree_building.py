import argparse
import sys
import os
import time_series_utils
from subsequence_tree_4 import KMedioidsSubsequenceTree
import pickle
import dill

parser = argparse.ArgumentParser(
    description='Build subsequence tree')
parser.add_argument('--sample_dir', required=True, type=str)
parser.add_argument('--distances_dir', required=True, type=str)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--n_samples', required=True, type=int)
parser.add_argument('--time_window', type=int, default=250)
parser.add_argument('--time_step', type=int, default=10)
parser.add_argument('--max_level', required=True, type=int)
parser.add_argument('--branching_factor', default=3, type=int)

args = parser.parse_args(sys.argv[1:])

sample_dir = args.sample_dir
distances_dir = args.distances_dir
output_dir = args.output_dir
dataset = args.dataset
n_samples = args.n_samples
time_window = args.time_window
time_step = args.time_step
max_level = args.max_level
branching_factor = args.branching_factor

sample_filename = 'sample_{0}_{1}_{2}_{3}.pkl'.format(dataset, n_samples,
                                                      time_window, time_step)
sample_path = os.path.join(sample_dir, sample_filename)
distances_filename = 'twed_{0}_{1}_{2}_{3}.pkl'.format(dataset, n_samples,
                                                       time_window, time_step)
distances_path = os.path.join(distances_dir, distances_filename)
output_filename = 'tree_{0}_{1}_{2}_{3}_{4}_kmedoids_bf{5}.dill'.format(dataset, n_samples,
                                                                        time_window, time_step,
                                                                        max_level, branching_factor)
output_path = os.path.join(output_dir, output_filename)

print('Opening samples file...')
with open(sample_path, 'rb') as f:
    sample = pickle.load(f)
print('DONE')
print('Opening distances file...')
with open(distances_path, 'rb') as f:
    distances_dict = pickle.load(f)
print('DONE')

print('Checking file correctnesss...')
sample_ids = [subsequence.id for subsequence in sample]
distances_ids = distances_dict['ids']
if sample_ids == distances_ids:
    print('Sample file corresponds to distances file')
else:
    print('Sample file doesn\'t correspond to distances file')
    exit()
print('DONE')

distances = distances_dict['distances']
affinities = -distances

print('Building tree...')

print('K MEDIOIDS APPROACH')
tree = KMedioidsSubsequenceTree(max_level, sample, distances,
                                time_window, time_step, branching_factor)

print('DONE')

print('Saving tree...')
with open( output_path, 'wb') as f:
    dill.dump(tree,  f)
print('DONE')

