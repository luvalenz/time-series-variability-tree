import numpy as np
import time

def relevance(retrieved_labels, relevant_label):
    return np.array(retrieved_labels) ==  relevant_label


def dcg(relevance):
    index = np.arange(len(relevance))
    discount = np.log2(index + 2)
    term = (2**relevance-1)/discount
    return np.cumsum(term)


def map(retrieved, relevant_label):
    relevants = relevance(retrieved, relevant_label)
    cumulative_relevants = np.cumsum(relevants)
    precision = cumulative_relevants/np.arange(1, len(cumulative_relevants) + 1)
    map_score = np.sum(precision*relevants)/np.sum(relevants)
    return map_score


def precision(retrieved, relevant_label, n):
    relevants = relevance(retrieved, relevant_label)
    cumulative_relevants = np.cumsum(relevants)
    precision_score = cumulative_relevants/np.arange(1, len(cumulative_relevants) + 1)
    length = len(precision_score)
    if n <= length:
        map_score = precision_score[:n]
    else :
        padding = n - length
        print(padding)
        map_score = np.concatenate((precision_score, -1*np.ones(padding)))
    return map_score


def ndcg(retrieved, relevant_label, n):
    rel_true = relevance(retrieved, relevant_label)
    rel_ideal = np.sort(rel_true)[::-1]
    dcg_score = dcg(rel_true)
    idcg_score = dcg(rel_ideal)
    ndcg_score = dcg_score/idcg_score
    length = len(ndcg_score)
    if n <= length:
        ndcg_score = ndcg_score[:n]
    else :
        padding = n - length
        print(padding)
        ndcg_score = np.concatenate((ndcg_score, -1*np.ones(padding)))
    return ndcg_score





class Timer:

    def __init__(self):
        self.elapsed_times = []
        self.current_start = None

    def start(self):
        self.current_start = time.time()

    def stop(self):
        current = time.time()
        elapsed = current - self.current_start
        self.elapsed_times.append(elapsed)
        self.print()

    def print(self):
        print(self.elapsed_times)
