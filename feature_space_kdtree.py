import pandas
import os
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
import time
from scoring_utils import ndcg
import sys
import dill
import numpy as np
import random
import glob2
import os
import pandas as pd


class FatsKDTreeSearcher:

    def __init__(self, data_frame):
        self.data = data_frame
        self.kdtree = KDTree(data_frame)

    @classmethod
    def build(cls, root_path):
        paths = glob2.glob(os.path.join(root_path, '**', '*.csv'))
        data_frames = [pd.read_csv(p) for p in paths]
        df = pd.concat(data_frames)
        return cls(df)








def build_kdtree(classes, n_components=None):
    root = '/home/lucas/tesis2'
    model_name = 'sequence_tree_20000samples_20levels'
    feature_space_path = os.path.join(root, 'macho_training_lightcurves/MACHO_ts2.csv'.format(model_name))
    df = pandas.read_csv(feature_space_path, index_col=0)
    index = list(df.index)
    db = df.values[:, :-1]
    numeric_labels = df.values[:, -1].astype(int)
    labels = [classes[label] for label in numeric_labels]
    if n_components is not None:
        pca = PCA(n_components)
        db = pca.fit_transform(db)
    kdtree = KDTree(db)
    new_df = pandas.DataFrame(db, index=index)
    return new_df, kdtree, labels


def build_mackenzie_kdtree(root, classes, data_file_name):
    feature_space_path = os.path.join(root, 'mackenzie_data/{0}'.format(data_file_name))
    df = pandas.read_csv(feature_space_path)
    df = df[~ df.iloc[:, 0].isnull()]
    index = df.values[:, -2]
    db = df.values[:, :-2]
    numeric_labels = df.values[:, -1].astype(int)
    labels = [classes[label] for label in numeric_labels]
    kdtree = KDTree(db)
    new_df = pandas.DataFrame(db, index=index)
    return new_df, kdtree, labels


def batch_queries(df, kdtree, labels, sample_ids, classes, n_results):
    results = {}
    times = []
    for lc_id in sample_ids:
        result, elapsed_time = query_ncdg(df, kdtree, labels, lc_id, n_results)
        results[lc_id] = result
        times.append(elapsed_time)
    return results, times


def query(df, kdtree, labels, query_id, n_results):
    start = time.time()
    lc_features = np.matrix(df.loc[query_id].values)
    print(lc_features)
    dist, ind = kdtree.query(lc_features, n_results)
    end = time.time()
    result_indices = list(ind.flatten())
    result_classes = [labels[index] for index in result_indices]
    elapsed = end - start
    return result_classes, elapsed


def query_ncdg(df, kdtree, labels, lightcurve_id, n_results):
    retrieved_classes, elapsed_time = query(df, kdtree, labels, lightcurve_id, n_results)
    position = list(df.index).index(lightcurve_id)
    class_ = labels[position]
    scores = ndcg(retrieved_classes, class_, n_results)
    return scores, elapsed_time


def sample_class(lcs_ids, labels, class_, samples_per_class):
    labels = np.array(labels)
    lcs_ids = np.array(lcs_ids)
    indices = np.where(labels == class_)[0]
    class_ids = lcs_ids[indices]
    return random.sample(class_ids.tolist(), samples_per_class)




if __name__ == '__main__':
    root = '/home/lucas/tesis2'
    #root = '/mnt/nas/GrimaRepo/luvalenz'
    samples_per_class = int(sys.argv[1])
    results_per_query = int(sys.argv[2])
    n_components = None
    # output_filename = 'test_outputs/fatsfeatures_pca{0}_{1}samples_per_class_{2}results_per_query.dill'.format(
    #                            n_components, samples_per_class, results_per_query)
    data_filename = 'affinity_s10t_w250_alpha_1.0_pool_max_scale_False_macho_part1of1.csv'
    output_filename = 'test_outputs/macfeatures_{0}samples_per_class_{1}results_per_query.dill'.format(samples_per_class, results_per_query)
    output_path = os.path.join(root, output_filename)
    classes = [None, 'BE', 'CEPH', 'EB', 'LPV', 'ML', 'NV', 'QSO', 'RRL']
    #df, kdtree, labels = build_kdtree(classes, n_components)
    df, kdtree, labels = build_mackenzie_kdtree(root, classes, data_filename)
    classes = classes[1:]
    results = {}
    times = {}
    for class_ in classes:
        lcs_ids = sample_class(list(df.index), labels, class_, samples_per_class)
        results[class_], times[class_] = batch_queries(df, kdtree, labels, lcs_ids, classes, results_per_query)
    test_output = {'results': results, 'times': times}
    with open(output_path, 'wb') as f:
        dill.dump(test_output, f)




