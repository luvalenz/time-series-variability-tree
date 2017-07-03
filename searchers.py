from scoring_utils import Timer, ndcg, map, precision
from scipy import stats
import numpy as np


class QueryResult:

    def __init__(self, target, ranking, times):
        self.target = target
        self.ranking = ranking
        self.times = times
        
    @property
    def preprocessed_ranking(self):
        ranking = np.array(self.ranking).flatten().tolist()
        pr = []
        for e in ranking:
           pr.append(e[3:] if e.startswith('lc_') else e)
        return pr

    def ndcg(self, class_table, length=20):
        target_class = class_table.loc[self.target, 'class']
        ranking_classes = class_table.loc[self.preprocessed_ranking[:length]]['class'].values
        return target_class, ndcg(ranking_classes, target_class, length)

    def map(self, class_table):
        target_class = class_table.loc[self.target, 'class']
        ranking_classes = class_table.loc[self.preprocessed_ranking]['class'].values
        return target_class, map(ranking_classes, target_class)

    def precision(self, class_table, length=20):
        target_class = class_table.loc[self.target, 'class']
        ranking_classes = class_table.loc[self.preprocessed_ranking[:length]]['class'].values
        return target_class, precision(ranking_classes, target_class, length)

    def kendall_tau(self, other_query_result):
        return stats.kendalltau(self.preprocessed_ranking, other_query_result.ranking)

    def __len__(self):
        return len(self.preprocessed_ranking)


class SubseuquenceSearcher:

    data_type = 'float64'

    def __init__(self, subseuquence_tree):
        self.st = subseuquence_tree

    def query(self, time_series):
        timer = Timer()
        ranking = self.st.make_query(time_series, timer).tolist()
        print(time_series.id)
        print(ranking[:5])
        times = timer.elapsed_times
        print(times)
        print('')
        return QueryResult(time_series.id, ranking, times)

