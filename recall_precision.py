import numpy as np
import matplotlib.pyplot as plt


class RecallPrecisionCurve:

    def __init__(self, query_class, sorted_result_classes):
        self.query_class = query_class
        self.relevant_indicator = sorted_result_classes == query_class

    @property
    def size(self):
        return len(self.relevant_indicator)

    @property
    def n_relevant(self):
        return sum(self.relevant_indicator)

    @property
    def average_precision(self):
        average = sum(self.relevant_indicator[i] * self.cutoff_precision(i)
                      for i in range(self.size))
        average /= self.n_relevant
        return average

    def cutoff_relevance(self, k ):
        return np.sum(self.relevant_indicator[:k])

    def cutoff_precision(self, k):
        return np.nan_to_num(self.cutoff_relevance(k)/k)

    def cutoff_recall(self, k):
        return np.nan_to_num(self.cutoff_relevance(k)/self.n_relevant)

    def get_original_curve(self):
        precision = [self.cutoff_precision(k) for k in range(self.size)]
        interpolated_p = [max(precision[k:]) for k in range(self.size)]
        recall = [self.cutoff_recall(k) for k in range(self.size)]
        return np.array(interpolated_p), np.array(recall)

    def get_interpolated_curve(self):
        precision, recall = self.get_original_curve()
        interpolated_p = [max(precision[k:]) for k in range(self.size)]
        return np.array(interpolated_p), np.array(recall)


if __name__ == '__main__':
    size = 1000
    n_classes = 4
    result = np.random.randint(0, n_classes, size)
    query = np.random.randint(0, n_classes)
    print(query)
    print(result)
    rpc = RecallPrecisionCurve(query, result)
    precision, recall = rpc.get_interpolated_curve()
    print(precision)
    print(recall)
    plt.plot(recall, precision)
    plt.show()




