import argparse
import sys
import os
import dill
import time_series_utils
import pandas as pd


parser = argparse.ArgumentParser(
    description='Build subsequence tree')
parser.add_argument('--results_path', required=True, type=str)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--class_table_path', default='', type=str)

args = parser.parse_args(sys.argv[1:])

class_table_path = args.class_table_path
results_path = args.results_path
output_dir = args.output_dir

class_table = None
print('reading class table...')
if class_table_path != '':
    class_table = time_series_utils.read_class_table(class_table_path)
print('DONE')


print('reading results...')
with open(results_path, 'rb') as f:
    results = dill.load(f)
print('DONE')

target_ids = []
target_classes = []
ndcgs = []
#maps = []
precisions = []
times = []
classes = []
ranking = []


print('Reordering results...')
for i, result in enumerate(results):
    print(i)
    target_id = result.target
    if class_table is not None:
        target_class, ndcg = result.ndcg(class_table)
        # target_class, map = result.map(class_table)
        target_class, precision = result.precision(class_table)
        target_classes.append(target_class)
        precisions.append(precision)
        ndcgs.append(ndcg)
    target_ids.append(target_id)
    times.append(result.times)
    ranking.append(result.ranking[:20])
print('DONE')

ndcg_df = pd.DataFrame(ndcgs)
times_df = pd.DataFrame(times)
ranking_df = pd.DataFrame(ranking)
# map_df = pd.DataFrame(maps)
precision_df = pd.DataFrame(precisions)

ndcg_df['id'] = target_ids
if class_table is not None:
    ndcg_df['class'] = target_classes

times_df['id'] = target_ids
if class_table is not None:
    times_df['class'] = target_classes

ranking_df['id'] = target_ids
if class_table is not None:
    times_df['class'] = target_classes

# map_df['id'] = target_ids
# map_df['class'] = target_classes

precision_df['id'] = target_ids
if class_table is not None:
    times_df['class'] = target_classes


results_basename = os.path.splitext(os.path.basename(results_path))[0]

if class_table is not None:
    ndcg_basename = 'ndcg__{0}.csv'.format(results_basename)
    precision_basename = 'precision__{0}.csv'.format(results_basename)
times_basename = 'times__{0}.csv'.format(results_basename)
ranking_basename = 'ranking__{0}.csv'.format(results_basename)
# map_basename = 'map__{0}.csv'.format(results_basename)

if class_table is not None:
    ndcg_output_path = os.path.join(output_dir, ndcg_basename)
    precision_output_path = os.path.join(output_dir, precision_basename)

times_output_path = os.path.join(output_dir, times_basename)
ranking_output_path = os.path.join(output_dir, ranking_basename)
# map_output_path = os.path.join(output_dir, map_basename)


print('Writing files...')
if class_table is not None:
    ndcg_df.to_csv(ndcg_output_path)
    precision_df.to_csv(precision_output_path)
times_df.to_csv(times_output_path)
ranking_df.to_csv(ranking_output_path)
# map_df.to_csv(map_output_path)

print('DONE')

