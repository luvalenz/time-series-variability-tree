from collections import Counter
import pydotplus as pydot
import numpy as np
from sklearn.cluster import KMeans


class FeatureTree:

    def __init__(self, max_level, db_features_list, k):
        self.max_level = max_level
        self.db_features_dict = {}
        self.db_features_ids = []
        self.graph = pydot.Dot(graph_type='graph')
        self.query_ts = None
        self.query_score_chart = None
        self.node_shortcuts = None
        self.weights = None
        self.d_matrix = None
        self._original_time_series_ids = None
        self._query_vector = None
        self.n_nodes = 0
        self.k = k
        id_ = 0
        for ts in db_features_list:
            ts._id = id_
            self.db_features_ids.append(ts._id)
            self.db_features_dict[ts._id] = ts
            id_ += 1
        self.root = Node(0, self.max_level, self.db_features_ids, affinities, None,
                         None, self.get_db_features_dict(),
                         self.get_next_node_id(), self.get_original_time_series_ids(), self.k)
        self._build_node_shorcuts()
        self._build_weights_vector()
        self._build_d_matrix()

    @property
    def n_features(self):
        return len(self.db_features_ids)

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
            q_norm = np.linalg.norm(q)
            self._query_vector = q_vector/ q_norm
        return self._query_vector

    def make_query(self, time_series):
        features = time_series.run_sliding_window()
        for node in self.node_shortcuts:
            node.n_query_features = 0
        self._query_vector = None
        for subsequence in features:
            self.root.add_query_subsequence(subsequence)

    def get_db_features_dict(self):
        def _get_db_features_dict():
            return self.db_features_dict
        return _get_db_features_dict

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

    def generate_graph(self):
        self.root.add_to_graph(None, self.graph)

    def _build_node_shorcuts(self):
        shortcut_dict = {}
        self.root.add_shortcut_to_dict(shortcut_dict)
        shortcut_list = [shortcut_dict[i] for i in range(self.n_nodes)]
        self.node_shortcuts = shortcut_list

    def _build_weights_vector(self):
        weights_list = [node.weight for node in self.node_shortcuts]
        self.weights = np.array(weights_list)

    def _build_d_matrix(self):
        d_list = []
        for node in self.node_shortcuts:
            d_list.append(node.d_vector)
        d_matrix = np.column_stack(d_list)
        d_norm = np.linalg.norm(d_matrix, axis=1)
        d_matrix = (d_matrix.T / d_norm).T
        d_matrix[d_matrix == np.inf] = 0
        self.d_matrix = np.nan_to_num(d_matrix)


class Node:

    def __init__(self, level, max_level, ids, affinities, center_id,
                 parent, db_features_dict_getter,
                 next_node_id_getter, original_time_series_ids_getter, k):
        self.level = level
        self.max_level = max_level
        self.features_ids = np.array(ids)
        self.center_id = center_id
        self.parent = parent
        self.get_db_features_dict = db_features_dict_getter
        self.get_original_time_series_ids_in_tree = original_time_series_ids_getter
        self._id = next_node_id_getter()
        self.n_query_features = 0
        self.k = k
        self.children = None
        self._inverted_file = None
        if level + 1 == max_level or len(ids) == 1:
            self._generate_inverted_files()
        else:
            self._generate_children(affinities, next_node_id_getter)

    def __str__(self):
        if self.children is None:
            return str(self.inverted_file())
        else:
            return super().__str__()

    @property
    def is_leaf(self):
        return self.children is None

    @property
    def inverted_file(self):
        if self.is_leaf:
            return self._inverted_file
        else:
            inverted_file = Counter()
            for child in self.children:
                inverted_file += child.inverted_file
            return inverted_file

    @property
    def n_original_time_series_in_node(self):
        return len(self.inverted_file)

    @property
    def n_original_time_series_in_tree(self):
        return len(self.get_original_time_series_ids_in_tree())

    @property
    def weight(self):
        return np.log(self.n_original_time_series_in_tree/
                      self.n_original_time_series_in_node)

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
        if self.n_query_features is None:
            return None
        return self.n_query_features*self.weight

    @property
    def d_vector(self):
        return self.weight*self.m_vector

    @property
    def center(self):
        features_dict = self.get_db_features_dict()
        return features_dict[self.center_id]

    def add_shortcut_to_dict(self, shortcut_dict):
        shortcut_dict[self._id] = self
        if not self.is_leaf:
            for child in self.children:
                child.add_shortcut_to_dict(shortcut_dict)

    def _generate_children(self, affinities, next_node_id_getter):
        print("---- GENERATING CHILDREN -----")
        print("affinities shape")
        print(affinities.shape)
        ap = KMeans(n_clusters=self.k)
        ap.fit(affinities)
        n_clusters = len(ap.cluster_centers_indices_)
        print("ids length")
        print(len(self.features_ids))
        print("center indices")
        print(ap.cluster_centers_indices_)
        cluster_centers_ids = self.features_ids[ap.cluster_centers_indices_]
        print("{0} clusters".format(n_clusters))
        labels = ap.labels_
        children = []
        i = 0
        for cluster_label, center_id in zip(range(n_clusters),
                                              cluster_centers_ids):
            indices = np.where(labels==cluster_label)[0]
            child_ids = self.features_ids[indices]
            child_affinities = affinities[indices][:, indices]
            print("child {0}".format(i))
            print("child indices")
            print(indices)
            print("child affinities")
            print(child_affinities)
            i += 1
            child = Node(self.level+1, self.max_level, child_ids,
                         child_affinities, center_id,
                         self, self.get_db_features_dict,
                         next_node_id_getter, self.get_original_time_series_ids_in_tree, self.k)
            children.append(child)
        self.children = children

    def add_query_subsequence(self, subsequence):
        self.n_query_features += 1
        if not self.is_leaf:
            distances = []
            for node in self.children:
                distance = time_series_twed(subsequence, node.center)
                distances.append(distance)
            nearest_child = self.children[np.argmin(distances)]
            nearest_child.add_query_subsequence(features)

    def _generate_inverted_files(self):
        self._inverted_file = Counter()
        db_features_dict = self.get_db_features_dict()
        for _id in self.features_ids:
            subsequence = db_features_dict[_id]
            original_time_series_id = subsequence.original_id
            counter = Counter({original_time_series_id: 1})
            self._inverted_file += counter

    def add_to_graph(self, parent_graph_node, graph):
        graph_node = pydot.Node(str(self))
        graph.add_node(graph_node)
        if parent_graph_node is not None:
            graph.add_edge(pydot.Edge(parent_graph_node,
                                      graph_node))
        if self.children is not None:
            for child in self.children:
                child.add_to_graph(graph_node, graph)