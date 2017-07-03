import numpy as np
from sklearn.cluster import AffinityPropagation
import os
import pandas as pd
import matplotlib.pyplot as plt


def get_distance_matrix(file_path):
    npz = np.load(file_path)
    return npz[npz.files[0]]


def run_affinity_propagation(affinities, preference):
    ap = AffinityPropagation(affinity='precomputed', preference=preference)
    ap.fit(affinities)
   # print(affinities == ap.affinity_matrix_)
    cluster_centers_indices = ap.cluster_centers_indices_
    n_clusters_ = len(cluster_centers_indices)
    return n_clusters_


def run_validation(affinities):
    x = np.arange(-10000, 10000, 1000)
    y = np.array([run_affinity_propagation(affinities, preference_factor)
                  for preference_factor in x])
#    reg = np.polyfit(np.log(x), np.log(y), 1)
    reg = None
    return x, y, reg


def plot_results(x, y, length):
    log_length = np.log(length)
    log_y_reg = np.poly1d(reg)(np.log(x))
    log_y_reg[log_y_reg > log_length] = log_length
    y_reg = np.exp(log_y_reg)
    plt.plot(x, y, '.')
    plt.plot(x, y_reg)
    plt.show()
    plt.plot(np.log(x), np.log(y), '.')
    plt.plot(np.log(x), log_y_reg)
    plt.show()



if __name__ == '__main__':
    #root = '/user/luvalenz/mackenzie_data/'
    #root = '/user/ivania/mackenzie_data/'
    root = '/home/lucas/Desktop/mackenzie_data'


    ms = []
    ns = []
    lengths = [1000, 2000, 3000, 4000, 20000]
    for i, l in enumerate(lengths):
        filename = 'twed_matrix_t_w=250_num{0}_macho.npz'.format(l)
        matrix_path = os.path.join(root, filename)
        distances = get_distance_matrix(matrix_path)
        affinities = -distances
        print("running validation...")
        x, y, reg = run_validation(affinities)
        print("DONE")
#        m, n = reg
#        print(m, n)
        plt.plot(x,y, '.')
        plt.show()
   # df = pd.DataFrame({'lengths': lengths, 'ms': ms, 'ns': ns})
   # df.to_csv('parameters_upto{0}.csv'.format(lengths[-1]), sep=';')
        #








