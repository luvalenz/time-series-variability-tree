from time_series import TimeSeriesOriginal
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
import os
import glob2 as glob


def get_nonvalid_rows(path):
    nonvalid_rows = 0
    with open(path, 'r') as f:
        for line in f:
            stripped_line = line[:-1].strip()
            n_cols = len(stripped_line.split(' '))
            if n_cols >= 3:
                break
            nonvalid_rows += 1
    return nonvalid_rows


def read_file(path):
    nonvalid_rows = get_nonvalid_rows(path)
    df = pd.read_csv(path, comment='#', sep=' ', header=None, skipinitialspace=True, skiprows=nonvalid_rows)
    not_nan_cols = np.where(~np.isnan(df.iloc[0]))[0]
    df = df[not_nan_cols]
    id_ = get_lightcurve_id(path)
    time = df.iloc[:, 0].values
    magnitude = df.iloc[:, 1].values
    not_nan = np.where(~np.logical_or(np.isnan(time), np.isnan(magnitude)))[0]
    time = time[not_nan]
    magnitude = magnitude[not_nan]
    ts = TimeSeriesOriginal(time, magnitude, id_)
    return ts


def read_files(paths):
    return (read_file(path) for path in paths)


def read_class_table(path):
    return pd.read_csv(path, sep=' ', index_col=0)


def stratified_sample(class_table, n_samples):
    X = class_table['path'].values
    y = class_table['class'].values
    if n_samples < len(y):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=n_samples, random_state=0)
        for train_index, test_index in sss.split(X, y):
            return X[test_index].tolist()

def nonstratified_sample(paths_file_path, n):
    with open(paths_file_path) as f:
        num_lines = sum(1 for line in f)
    sample_indices = np.sort(np.random.choice(num_lines, n, replace=False))
    pointer = 0
    sample = []
    with open(paths_file_path) as f:
        for i, line in enumerate(f):
            if pointer == n:
                break
            if i == sample_indices[pointer]:
                sample.append(line[:-1])
                pointer += 1
    return sample


def get_lightcurve_id(fp):
    basename = os.path.basename(fp)
    filename = '.'.join(basename.split('.')[:-1])
    if filename.startswith('lc_'):
        filename = filename[3:]
    if filename.endswith('.B') or filename.endswith('.R'):
        filename = filename[:-2]
    return filename


def add_paths_to_class_table(class_table, paths):
    index = class_table.index
    class_table['path'] = pd.Series(np.zeros_like(index.values), index=index)
    for p in paths:
        id_ = get_lightcurve_id(p)
        if id_ in class_table.index:
            class_table.loc[id_, 'path'] = p


def read_files(file_paths, part=None, n_parts=None):
    print('Getting chunk of data...')
    if part is not None:
        chunk_length = int(len(file_paths) // n_parts)
        if part == n_parts - 1:
            file_paths = file_paths[part*chunk_length:]
        else:
            file_paths = file_paths[part*chunk_length:(part+1)*chunk_length]
    for path in file_paths:
        if path.endswith('\n'):
            path = path[:-1]
        if os.path.exists(path):
            yield read_file(path)
        else:
            print('doesn\'t exist')


