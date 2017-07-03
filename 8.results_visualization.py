import argparse
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import pickle



def plot_by_classes(values, classes, path, class_names, plot_title, y_label):
    unique_classes = list(set(classes))
    values_by_class = [values[classes == c].tolist() for c in unique_classes]
    labels = [class_names[i] for i in unique_classes]
    plt.clf()
    plt.gcf().subplots_adjust(bottom=0.30)
    plt.boxplot(values_by_class, showmeans=True)
    plt.xticks(unique_classes, labels, rotation=75)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.savefig(path)

parser = argparse.ArgumentParser(
    description='Build subsequence tree')
parser.add_argument('--results_path', required=True, type=str)
parser.add_argument('--times_path', required=True, type=str)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--class_names_path', required=True, type=str)
parser.add_argument('--plot_title', default='', type=str)

args = parser.parse_args(sys.argv[1:])

results_path = args.results_path
times_path = args.times_path
output_dir = args.output_dir
class_names_path = args.class_names_path
plot_title = args.plot_title


with open(class_names_path, 'rb') as f:
    class_names = pickle.load(f)

print(class_names)

ndcg_basename = os.path.splitext(os.path.basename(results_path))[0]
times_basename = os.path.splitext(os.path.basename(times_path))[0]


results = pd.read_csv(results_path, index_col=0)
times = pd.read_csv(times_path, index_col=0)

classes = results['class'].values

for i in [1, 5, 10, 15]:
    ndcg_path = os.path.join(output_dir, '{0}_nDCGat{1}.jpg'.format(ndcg_basename, i))
    ndcg = results[str(i -1)].values
    y_label = 'nDCG@{}'.format(i)
    ndcg_plot_title = y_label + ' ' + plot_title
    plot_by_classes(ndcg, classes, ndcg_path, class_names, ndcg_plot_title, y_label)

times_plot_title = 'Execution time ' + plot_title
y_label = 'time [s]'
times_output_path = os.path.join(output_dir, '{0}_times.jpg'.format(times_basename))

total_times = times.drop(['id', 'class'], axis=1).values.sum(axis=1)
plot_by_classes(total_times, classes, times_output_path, class_names, times_plot_title, y_label)
