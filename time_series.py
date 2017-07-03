import numpy as np
import random


class TimeSeries:

    def __init__(self, time, magnitude, id_):
        self.time = time
        self.magnitude = magnitude
        self.id = id_

    def __str__(self):
        return self.id


class TimeSeriesOriginal(TimeSeries):

    def __init__(self, time, magnitude, id_,
                 semi_standardize=False,
                 standardize_std=True):
        super().__init__(time, magnitude, id_)
        if semi_standardize or standardize_std:
            self.standardize_mean()
        if standardize_std:
            self.standardize_std()

    @property
    def total_time(self):
        return self.time[-1] - self.time[0]

    def standardize_mean(self):
        mean = self.magnitude.mean()
        self.magnitude -= mean

    def standardize_std(self):
        std = self.magnitude.std()
        self.magnitude /= std

    def run_sliding_window(self, time_window, time_step):
        t_f = self.time[-1]
        last_pos_index = np.where(self.time > t_f - time_window)[0][0]
        last_pos_time = self.time[last_pos_index]
        start_time = self.time[0]
        while start_time < last_pos_time:
            yield self._get_subsequence(start_time, time_window)
            start_time += time_step
            start_index = np.where(self.time >= start_time)[0][0]
            start_time = self.time[start_index]

    def get_random_subsequences(self, n, time_window):
        return (self.get_random_subsequence(time_window) for i in range(n))

    def get_random_subsequence(self, time_window):
        print('taking subsequence')
        print(self.id)
        print(type(self.time))
        print(self.time)
        print(type(self.magnitude))
        print(self.magnitude)
        t_f = self.time[-1]
        t_0 = self.time[0]
        if time_window > t_f - t_0 :
            raise AttributeError('Time window larger than Time Series. id: {0} '.format(self.id))
        last_pos_index = np.where(self.time > t_f - time_window)[0][0]
        start_index = random.choice(range(last_pos_index))
        start_value = self.time[start_index]
        return self._get_subsequence(start_value, time_window)

    def _get_subsequence(self, start_time, time_window):
        end_time = start_time + time_window
        indices = np.where(np.logical_and(start_time <= self.time, self.time < end_time))[0]
        sub_time = self.time[indices]
        sub_mag = self.magnitude[indices]
        subsequence_id = '{0}.{1}.{2}'.format(self.id, start_time, time_window)
        subsequence = TimeSeriesSubsequence(sub_time, sub_mag, subsequence_id, self.id)
        return subsequence


class TimeSeriesSubsequence(TimeSeries):

    def __init__(self, time, magnitude, id_, original_id):
        super().__init__(time, magnitude, id_)
        self.original_id = original_id
        self.zeroify_time()

    def zeroify_time(self):
        t0 = self.time[0]
        self.time -= t0

