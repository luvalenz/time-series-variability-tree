import numpy as np
from sklearn.cluster import AffinityPropagation
#import pydotplus as pydot
from collections import Counter
from distance_utils import time_series_twed
import pandas as pd
from scipy.spatial.distance import squareform


class BottomUpSubsequenceTree:

    def __init__(self, max_level, prototype_subsequences_list,
                 affinities, db_time_series,
                 time_window, time_step, weighted=True, max_branching_factor=20):
        self.time_window = time_window
        self.time_step = time_step
        self.max_level = max_level
        self.max_branching_factor = max_branching_factor
        #self.graph = pydot.Dot(graph_type='graph')
        self.query_ts = None
        self.query_score_chart = None
        self.node_shortcuts = None
        self.weights = None
        self.d_data_frame = None
        self._original_time_series_ids = None
        self._query_vector = None
        self.n_nodes = 0
        self._weighted = weighted
        prototype_subsequences = np.array(prototype_subsequences_list)
        self._build_tree(affinities, prototype_subsequences)
        self._populate_tree(db_time_series)
        self._build_node_shorcuts()
        self._build_weights_vector()
        self._build_d_data_frame()

    @property
    def n_subsequences(self):
        return len(self.db_subsequences_dict)

    @property
    def original_time_series_ids(self):
        if self._original_time_series_ids is None:
            self._original_time_series_ids = list(self.root.inverted_file)
        return self._original_time_series_ids

    @property
    def n_original_time_series(self):
        return len(self.original_time_series_ids)

    @property
    def query_vector(self):
        if self._query_vector is None:
            q_vector = np.array([node.q for node in self.node_shortcuts])
            q_norm = np.linalg.norm(q_vector)
            self._query_vector = q_vector / q_norm
        return self._query_vector

    @property
    def _queried_time_series_ids(self):
        return list(set().union(*self._queried_time_series_ids_iterator()))

    def prune(self):
        self.root.prune_branch(1, self.max_level)

    def _queried_time_series_ids_iterator(self):
        for node in self.node_shortcuts:
            if node.is_leaf and node.n_query_subsequences > 0:
                yield node.inverted_file.keys()

    def make_query(self, time_series, timer=None):
        if timer is not None:
            timer.start()
        subsequences = time_series.run_sliding_window(self.time_window, self.time_step)
        if timer is not None:
            timer.stop()
            timer.start()
        for node in self.node_shortcuts:
            node.n_query_subsequences = 0
        if timer is not None:
            timer.stop()
            timer.start()
        self._query_vector = None
        for subsequence in subsequences:
            self.root.add_query_subsequence(subsequence)
        if timer is not None:
            timer.stop()
            timer.start()
        not_zero_node_ids = np.where(self.query_vector != 0)[0]
        not_zero_query_vector = self.query_vector[not_zero_node_ids]
        not_zero_ts_ids = self._queried_time_series_ids
        not_zero_d_dataframe = self.d_data_frame.loc[not_zero_ts_ids, not_zero_node_ids]
        if timer is not None:
            timer.stop()
            timer.start()
        score = -np.sum(not_zero_query_vector*not_zero_d_dataframe.values, axis=1)
        #score = 2-2*score
        if timer is not None:
            timer.stop()
            timer.start()
        order = np.argsort(score)
        result = not_zero_d_dataframe.index.values[order]
        if timer is not None:
            timer.stop()
        return result

    def get_db_subsequences_dict(self):
        def _get_db_subsequences_dict():
            return self.db_subsequences_dict
        return _get_db_subsequences_dict

    def get_next_node_id(self):
        def _get_next_node_id():
            n_nodes = self.n_nodes
            self.n_nodes += 1
            return n_nodes
        return _get_next_node_id

    def get_original_time_series_ids(self):
        def _get_original_time_series_ids():
            return self.original_time_series_ids
        return _get_original_time_series_ids

    # def save_graph(self):
    #     self.generate_graph()
    #     self.graph.write_png('graph.png')
    #
    # def generate_graph(self):
    #     self.root.add_to_graph(None, self.graph)

    def _build_tree(self, affinities, subsequences):
        print('Building layer 0')
        center_indices, labels = self.run_affinity_propagation(affinities, True)
        centers = subsequences[center_indices]
        affinities = affinities[center_indices][:, center_indices]
        nodes = self._build_leaves(centers)
        print("{} nodes".format(len(nodes)))
        levels = 1
        while len(nodes) > self.max_branching_factor:
            print('Building layer {0}'.format(levels))
            center_indices, labels = self.run_affinity_propagation(affinities, False)
            centers = centers[center_indices]
            affinities = affinities[center_indices][:, center_indices]
            nodes = self._build_layer(nodes, centers, labels)
            print("{} nodes".format(len(nodes)))
            levels += 1
        if len(nodes) == 1:
            self.root = nodes[0]
        else:
            self.root = Node(None, nodes, self.get_next_node_id(), self.get_original_time_series_ids())
        if levels > self.max_level:
            self.prune()


    def _build_leaves(self, leaf_centers):
        return [Node(center, None, self.get_next_node_id(), self.get_original_time_series_ids())
                for center in leaf_centers]

    def _build_layer(self, lower_nodes, upper_centers, labels):
        lower_nodes = np.array(lower_nodes)
        nodes = []
        for i, center in enumerate(upper_centers):
            children = lower_nodes[np.where(labels == i)[0]]
            if len(children) == 1:
                nodes.append(children[0])
            else:
                nodes.append(Node(center, children, self.get_next_node_id(), self.get_original_time_series_ids()))
        return nodes

    def _populate_tree(self, db_time_series):
        print("populating tree")
        print('time window')
        print(self.time_window)
        print('time step')
        print(self.time_step)
        print(type(db_time_series))
        print(db_time_series)
        for i, ts in enumerate(db_time_series):
            print(ts)
            for subsequence in ts.run_sliding_window(self.time_window, self.time_step):
                self._add_subsequence(subsequence)
            print("{0} time series added".format(i))

    def _build_node_shorcuts(self, just_leaves=False):
        shortcut_dict = {}
        self.root.add_shortcut_to_dict(shortcut_dict)
        shortcut_list = [v for v in shortcut_dict.values()
                         if not just_leaves or v.is_leaf]
        self.node_shortcuts = shortcut_list

    def _build_weights_vector(self):
        weights_list = [node.weight for node in self.node_shortcuts]
        self.weights = np.array(weights_list)

    def _build_d_data_frame(self, just_leaves=False):
        d_list = [node.d_vector for node in self.node_shortcuts]
        d_matrix = np.column_stack(d_list)
        d_norm = np.linalg.norm(d_matrix, axis=1)
        d_matrix = (d_matrix.T / d_norm).T
        d_matrix[d_matrix == np.inf] = 0
        self.d_data_frame = pd.DataFrame(np.nan_to_num(d_matrix),
                                       index=self.original_time_series_ids)

    def _add_subsequence(self, subsequence):
        self.root.add_db_subsequence(subsequence)

    def calculate_inverted_files(self):
        return self.root.inverted_file

    def run_affinity_propagation(self, affinities, leaves):
        affinities_list = squareform(affinities)
        preference = np.median(affinities_list)
        branching_factor = np.inf
        while branching_factor > self.max_branching_factor:
            ap = AffinityPropagation(affinity='precomputed')
            ap.preference = preference
            ap.fit(affinities)
            if leaves:
                branching_factor = 0
            else:
               # branching_factor = max(Counter(ap.labels_).values())
                branching_factor = affinities.shape[0]//len(ap.cluster_centers_indices_)
            preference += (np.max(affinities_list) - np.min(affinities_list))/500
            print('branching factor = {0}'.format(branching_factor))
        print('labels')
        print(Counter(ap.labels_))
        return ap.cluster_centers_indices_, ap.labels_


class Node:

    def __init__(self, center, children,
                 next_node_id_getter, original_time_series_ids_getter,
                 weighted=True):
        self._weighted = weighted
        self.center = center
        self.get_original_time_series_ids_in_tree = original_time_series_ids_getter
        self._id = next_node_id_getter()
        self.n_query_subsequences = 0
        self.children = children
        self._inverted_file = None
        if children is None:
            self._generate_inverted_file()

    @property
    def is_leaf(self):
        return self.children is None

    @property
    def inverted_file(self):
        if self._inverted_file is None:
            inverted_file = Counter()
            for child in self.children:
                inverted_file += child.inverted_file
            self._inverted_file = inverted_file
        return self._inverted_file

    @property
    def n_original_time_series_in_node(self):
        return len(self.inverted_file)

    @property
    def n_original_time_series_in_tree(self):
        return len(self.get_original_time_series_ids_in_tree())

    @property
    def weight(self):
        w = 0
        if self.n_original_time_series_in_node != 0:
            w = np.log(self.n_original_time_series_in_tree/
                       self.n_original_time_series_in_node)
        try:
            if not self._weighted:
                w = 1
        except AttributeError:
            print("Attribute Error caught")
            print("weight = {0}".format(w))
        return w

    @property
    def m_vector(self):
        m = np.zeros(self.n_original_time_series_in_tree)
        ids = self.get_original_time_series_ids_in_tree()
        for key, value in self.inverted_file.items():
            index = ids.index(key)
            m[index] = value
        return m

    @property
    def q(self):
        if self.n_query_subsequences is None:
            return None
        return self.n_query_subsequences*self.weight

    @property
    def d_vector(self):
        return self.weight*self.m_vector

    def add_shortcut_to_dict(self, shortcut_dict):
        shortcut_dict[self._id] = self
        if not self.is_leaf:
            for child in self.children:
                child.add_shortcut_to_dict(shortcut_dict)

    def add_query_subsequence(self, subsequence):
        self.n_query_subsequences += 1
        if not self.is_leaf:
            distances = [time_series_twed(subsequence, node.center)
                        for node in self.children]
            nearest_child = self.children[np.argmin(distances)]
            nearest_child.add_query_subsequence(subsequence)

    def add_db_subsequence(self, subsequence, level=0):
        if self.is_leaf:
            counter = Counter({subsequence.original_id: 1})
            self._inverted_file += counter
        else:
            distances = [time_series_twed(subsequence, node.center)
                        for node in self.children]
            nearest_child = self.children[np.argmin(distances)]
            nearest_child.add_db_subsequence(subsequence, level+1)

    def _generate_inverted_file(self):
        self._inverted_file = Counter()

    def prune_branch(self, level, max_level):
        if level == max_level:
            self.children = None
            self._generate_inverted_file()
        elif self.children is not None:
            for child in self.children:
                child.prune_branch(level + 1, max_level)
