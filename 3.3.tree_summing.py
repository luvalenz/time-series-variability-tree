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


def get_partial_trees(input_tree_path, n_parts):
    for part in range(n_parts):
        part_filename = input_tree_path + '.part{}of{}'.format(part, n_parts)
        print(part_filename)
        with open(part_filename, 'rb') as f:
            partial_tree = dill.load(f)
        yield partial_tree

parser = argparse.ArgumentParser(
    description='Build subsequence tree')

parser.add_argument('--input_tree_path', default='', type=str)
parser.add_argument('--n_parts', required=True, type=int)


args = parser.parse_args(sys.argv[1:])

input_tree_path = args.input_tree_path
n_parts = args.n_parts


with open(input_tree_path, 'rb') as f:
    tree = dill.load(f)

partial_trees = get_partial_trees(input_tree_path, n_parts)

tree.populate_from_tree_sum(partial_trees)

print('DONE')

print('Saving tree...')
with open( input_tree_path + '.populated', 'wb') as f:
    dill.dump(tree,  f, protocol=4)
print('DONE')

