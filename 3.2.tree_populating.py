import argparse
import sys
import os
import time_series_utils
from subsequence_tree import SubsequenceTree
from subsequence_tree_2 import BottomUpSubsequenceTree
from subsequence_tree_3 import BottomUpSubsequenceTree as Tree3
from subsequence_tree_4 import KMedioidsSubsequenceTree
import pickle
import dill

parser = argparse.ArgumentParser(
    description='Build subsequence tree')
parser.add_argument('--dataset_root', default='', type=str)
parser.add_argument('--input_paths_file', default='', type=str)
parser.add_argument('--class_table_path', default='', type=str)
parser.add_argument('--tree_path', required=True, type=str)
parser.add_argument('--part', required=True, type=int)
parser.add_argument('--n_parts', required=True, type=int)


args = parser.parse_args(sys.argv[1:])

dataset_root = args.dataset_root
input_paths_file = args.input_paths_file
class_table_path = args.class_table_path
tree_path = args.tree_path
part = args.part
n_parts = args.n_parts

print('part{}  of {}'.format(part, n_parts))


if input_paths_file != '':
    print('Reading file paths')
    with open(input_paths_file, 'r') as f:
        lightcurves_paths = f.readlines()
    print('DONE')

elif class_table_path  != '':
    class_table = time_series_utils.read_class_table(class_table_path)
    lightcurves_paths = class_table['path'].values
    lightcurves_paths = [os.path.join(dataset_root, p) for p in lightcurves_paths]

print('Reading dataset...')
dataset = time_series_utils.read_files(lightcurves_paths, part, n_parts)
print('DONE')



with open(tree_path, 'rb') as f:
    tree = dill.load(f)


dataset = (lc for lc in dataset if lc.total_time >= tree.time_window)

output_path = tree_path + '.part{}of{}'.format(part, n_parts)

print(output_path)

tree.populate(dataset)

print('DONE')

print('Saving tree...')
with open(output_path, 'wb') as f:
    dill.dump(tree,  f)
print('DONE')

