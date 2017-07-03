import os
import sys


def execute_pipleline(n_prototipes, window_size, step, max_level, standardization, weight):
    call1 = 'python 1.sample.py lightcurves.R.txt {0} {1} {2} {3}'.format(n_prototipes, standardization, window_size, step)
    call2 = 'python 2.twed.py lightcurves.R.txt {0} {1} {2} {3}'.format(n_prototipes, standardization, window_size, step)
    call3 = 'python 3.tree_building.py lightcurves.R.txt {0} {1} {2} {3} {4} {5}'.format(n_prototipes, standardization,
                                                                                         weight, window_size, step, max_level)
    standardization_dict = {'std':'semistdFalse_stdTrue', 'semi': 'semistdTrue_stdFalse', 'not': 'semistdFalse_stdFalse'}
    standardization_str = standardization_dict[standardization]
    model_name = 'sequence_tree_lightcurves.R.txt_{0}samples_{1}_20levels_{2}_{3}'.format(n_prototipes, standardization_str,
                                                                                          window_size, step)
    call4 = 'python 4.ndcg_test.py {0} 100 20'.format(model_name)
    print('RUNNING 1.SAMPLE')
    os.system(call1)
    print('RUNNING 2.TWED')
    os.system(call2)
    print('RUNNING 3.TREE_BUILDING')
    os.system(call3)
    print('RUNNING 4.NDCG_TEST')
    os.system(call4)
    print('DONE')

if __name__ == '__main__':
    n = int(sys.argv[1])
    window = int(sys.argv[2])
    step = int(sys.argv[3])
    max_level = sys.argv[4]
    standardization = sys.argv[5]
    weight = sys.argv[6]
    execute_pipleline(n, window, step, max_level, standardization, weight)

