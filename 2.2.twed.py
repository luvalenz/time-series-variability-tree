import argparse
import sys
import os
from distance_utils import time_series_twed
from scipy.spatial.distance import squareform
import itertools
import pickle



parser = argparse.ArgumentParser(
    description='Calculate distances of subsequences')
parser.add_argument('--input_dir', required=True, type=str)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--n_samples', required=True, type=int)
parser.add_argument('--time_window', type=int, default=250)
parser.add_argument('--time_step', type=int, default=10)
parser.add_argument('--n_parts', type=int)

args = parser.parse_args(sys.argv[1:])

input_dir = args.input_dir
output_dir = args.output_dir
dataset = args.dataset
n_samples = args.n_samples
time_window = args.time_window
time_step = args.time_step
n_parts = args.n_parts


input_filename = 'sample_{0}_{1}_{2}_{3}.pkl'.format(dataset, n_samples,
                                                      time_window, time_step)
output_filename = 'twed_{0}_{1}_{2}_{3}.pkl'.format(dataset, n_samples,
                                                      time_window, time_step)
input_path = os.path.join(input_dir, input_filename)
output_path = os.path.join(output_dir, output_filename)

print('Loading data...')
with open(input_path, 'rb') as f:
    subsequences = pickle.load(f)
print('DONE')


print('getting ids...')
ids = [s.id for s in subsequences]
print('DONE')


distances = []

print('building distance matrix...')
for i in range(n_parts):
    input_path = output_path + '.part{}of{}'.format(i, n_parts)
    with open(input_path, 'rb') as f:
        distances += pickle.load(f)

distance_matrix = squareform(distances)
print('DONE')


output = {'ids': ids, 'distances': distance_matrix}

print('Writing output...')
with open(output_path, 'wb') as f:
    pickle.dump(output, f, protocol=4)
print('DONE')