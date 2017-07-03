import argparse
import sys
import os
import pickle
import time_series_utils
import random


parser = argparse.ArgumentParser(
    description='Get samples from lightcurves.')
parser.add_argument('--dataset_root', default='', type=str)
parser.add_argument('--input_paths_file', default='', type=str)
parser.add_argument('--class_table_path', default='', type=str)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--n_samples', required=True, type=int)
parser.add_argument('--time_window', type=int, default=250)
parser.add_argument('--time_step', type=int, default=10)

args = parser.parse_args(sys.argv[1:])

dataset_root = args.dataset_root
input_paths_file = args.input_paths_file
class_table_path = args.class_table_path
output_dir = args.output_dir
dataset = args.dataset
n_samples = args.n_samples
time_window = args.time_window
time_step = args.time_step

output_filename = 'sample_{0}_{1}_{2}_{3}.pkl'.format(dataset, n_samples,
                                                      time_window, time_step)
output_path = os.path.join(output_dir, output_filename)


print('Sampling lightcurves...')
extended_n_samples = int(1.2*n_samples)
if class_table_path == '':
    relative_paths_sample = time_series_utils.nonstratified_sample(input_paths_file, extended_n_samples)

else:
    class_table = time_series_utils.read_class_table(class_table_path)
    if extended_n_samples > len(class_table):
        n_repetitions = n_samples // len(class_table) + 1
        relative_paths_sample = n_repetitions * class_table['path'].tolist()
        random.shuffle(relative_paths_sample)
        relative_paths_sample[:n_samples]
    else:
        relative_paths_sample = time_series_utils.stratified_sample(class_table, extended_n_samples)

print('DONE')

abs_paths = (os.path.join(dataset_root, path) for path in relative_paths_sample)

print('Opening lightcurves...')
lightcurves_sample = (time_series_utils.read_file(path) for path in abs_paths)
lightcurves_sample = (lc for lc in lightcurves_sample if lc.total_time >= time_window)
print('DONE')

print('Getting subsequences...')
subsequences_sample = [lc.get_random_subsequence(time_window)
                       for lc in lightcurves_sample][:n_samples]

print('Sample\'s actual length = {0}'.format(len(subsequences_sample)))
print('DONE')

with open(output_path, 'wb') as f:
    pickle.dump(subsequences_sample, f, protocol=2)





