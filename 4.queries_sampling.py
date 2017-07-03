import argparse
import sys
import time_series_utils
import dill
import pickle
import numpy as np
import os


def query_nonlabeled_sample(paths_file_path, n):
    with open(paths_file_path) as f:
        num_lines = sum(1 for line in f)
    sample_indices = np.sort(np.random.choice(num_lines, n, replace=False))
    pointer = 0
    sample = []
    with open(paths_file_path) as f:
        for i, line in enumerate(f):
            if pointer == n:
                break
            if i == sample_indices[pointer]:
                sample.append(line[:-1])
                pointer += 1
    return sample


def query_labeled_sample(classes_df, n_per_class):
    classes = set(classes_df['class'])
    sample = []
    for class_ in classes:
        paths = classes_df[classes_df['class'] == class_]['path'].tolist()
        try:
            class_sample = np.random.choice(paths, n_per_class, replace=False)
        except ValueError:
            class_sample = paths
        sample += list(class_sample)
    return sample


parser = argparse.ArgumentParser(
    description='Build subsequence tree')
parser.add_argument('--class_table_path', default='',  type=str)
parser.add_argument('--paths_list_path', default='',  type=str)
parser.add_argument('--n_queries', required=True, type=int)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--name', required=True, type=str)
parser.add_argument('--dataset', required=True, type=str)

args = parser.parse_args(sys.argv[1:])


class_table_path = args.class_table_path
paths_list_path = args.paths_list_path
n_queries= args.n_queries
output_dir = args.output_dir
name = args.name
dataset = args.dataset


if class_table_path == '':
    paths = query_nonlabeled_sample(paths_list_path, n_queries)
    sample_type = 'nonlabeled'
else:
    class_table = time_series_utils.read_class_table(class_table_path)
    paths = query_labeled_sample(class_table, n_queries)
    sample_type = 'labeled'

#lcs = time_series_utils.read_files(paths)

basename = 'querysample_{0}_{1}_{2}.dill'.format(name, sample_type, n_queries)
output_path = os.path.join(output_dir, basename)

print(paths)

with open(output_path, 'wb') as f:
    dill.dump(paths, f, protocol=2)

def top2(path):
    with open(path, 'rb') as f:
        obj = dill.load(f)
    new_path = path.replace('vt_data', 'fats_data')
    print(obj)
    with open(new_path, 'wb') as f:
        pickle.dump(obj, f, protocol=0)

import pickle


def get_lightcurve_path(lc_id, dataset, band='B'):
    if dataset == 'macho':
        field, tile, seq = lc_id.split('.')
        basename = 'lc_{0}.{1}.{2}.{3}.mjd'.format(field, tile, seq, band)
        return os.path.join('F_{0}'.format(field), tile, basename)
    elif dataset == 'ogle':
        try:
            ogle, region, class_, n = lc_id.split('-')
            basename = 'OGLE-{0}-{1}-{2}.dat'.format(region, class_, n)
            return os.path.join(region.lower(), class_.lower(), basename)
        except ValueError:
            return ''
    elif dataset == 'kepler':
        basename = '{:09d}.dat'.format(int(lc_id))
        return os.path.join(basename[:4], basename)
    else:
        exit(-1)


def ids_to_paths(path, dataset):
    with open(path, 'rb') as f:
        ids_list = pickle.load(f)
    paths_list = [get_lightcurve_path(lc_id, dataset) for lc_id in ids_list]
    with open(path, 'wb') as f:
        pickle.dump(paths_list, f)


paths = ['vt_data/macho/querysample_macho_labeled_100.pkl', 'vt_data/macho2/querysample_macho2_labeled_500.pkl', 'vt_data/kepler/querysample_kepler_labeled_500.pkl', 'vt_data/ogle/querysample_ogle_labeled_100.pkl']

for p in paths:
    ids_to_paths(paths, dataset)