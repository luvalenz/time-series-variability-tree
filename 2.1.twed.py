import argparse
import sys
import os
import pickle
from scipy.misc import comb
import itertools
from distance_utils import time_series_twed
import time

def chunks(length, n_chunks, chunk):
    """Yield successive n-sized chunks from l."""
    chunk_size = int(length // n_chunks)
    i = chunk*chunk_size
    j = i + chunk_size
    if chunk == n_chunks - 1:
        j = int(length)
    return i, j


parser = argparse.ArgumentParser(
    description='Calculate distances of subsequences')
parser.add_argument('--input_dir', required=True, type=str)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--n_samples', required=True, type=int)
parser.add_argument('--time_window', type=int, default=250)
parser.add_argument('--time_step', type=int, default=10)
parser.add_argument('--part', type=int)
parser.add_argument('--n_parts', type=int)

args = parser.parse_args(sys.argv[1:])

input_dir = args.input_dir
output_dir = args.output_dir
dataset = args.dataset
n_samples = args.n_samples
time_window = args.time_window
time_step = args.time_step
part = args.part
n_parts = args.n_parts

input_filename = 'sample_{0}_{1}_{2}_{3}.pkl'.format(dataset, n_samples,
                                                      time_window, time_step)
input_path = os.path.join(input_dir, input_filename)
output_filename = 'twed_{0}_{1}_{2}_{3}.pkl.part{4}of{5}'.format(dataset, n_samples,
                                                                 time_window, time_step,
                                                                 part, n_parts)

output_path = os.path.join(output_dir, output_filename)

print('Loading data...')
with open(input_path, 'rb') as f:
    subsequences = pickle.load(f)
print('DONE')

n_samples = len(subsequences)
n_dist = comb(n_samples, 2)
begin, end = chunks(n_dist, n_parts, part)
print(begin)
print(end)

print('Calculating distances...')

distances = []
t = time.time()
for i, j in itertools.islice(itertools.combinations(subsequences, 2), begin, end):
    print('Calculating distance between {} and {}'.format(i, j))
    distances.append(time_series_twed(i, j))
    print('elapsed = {}'.format(time.time() - t))
    t = time.time()

print('Writing output...')
print(output_path)
with open(output_path, 'wb') as f:
    pickle.dump(distances, f, protocol=4)
print('DONE')