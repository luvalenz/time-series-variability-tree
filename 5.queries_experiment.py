import argparse
import sys
import os
import dill
import time_series_utils
from searchers import SubseuquenceSearcher


parser = argparse.ArgumentParser(
    description='Build subsequence tree')
parser.add_argument('--tree_path', required=True, type=str)
parser.add_argument('--query_sample_path', required=True, type=str)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--dataset_root', required=True, type=str)

args = parser.parse_args(sys.argv[1:])

tree_path = args.tree_path
query_sample_path = args.query_sample_path
output_dir = args.output_dir
dataset_root = args.dataset_root


print('Loading tree...')
with open(tree_path, 'rb') as f:
    tree = dill.load(f)
print('DONE')


with open(query_sample_path, 'rb') as f:
    query_sample = dill.load(f)

paths = (os.path.join(dataset_root, p) for p in query_sample)
lcs = time_series_utils.read_files(paths)

results = []
for lc in lcs:
    searcher = SubseuquenceSearcher(tree)
    query_result = searcher.query(lc)
    results.append(query_result)


tree_name = os.path.splitext(os.path.basename(tree_path))[0]
sample_name = os.path.splitext(os.path.basename(query_sample_path))[0]
basename = '{0}__{1}.dill'.format(tree_name, sample_name)

output_path = os.path.join(output_dir, basename)

with open(output_path, 'wb') as f:
    dill.dump(results, f)
