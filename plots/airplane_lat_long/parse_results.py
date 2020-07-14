import os
import pandas as pd
import numpy as np


def pretty_print_latex(matrix):
    """Prints matrix in a format that can be pasted into LaTex"""
    row = ""
    for loss in matrix[[3, 5, 2, 4, 6, 7, 10, 11]]:
        if loss < 1e-2: # use scientific notation
            row += (" {0:.1e}          &".format(float(loss))).replace('-0', '-')
        else:
            row += " {0:.3f}           &".format(float(loss))
    print(row)

def parse_filename(file):
    name = file.split('results')[-1]
    if '100' in name:
        return '100'
    if '10' in name:
        return '10'
    if '1' in name:
        return '1'
    return

for model in ['odenet', 'densenet', 'learnedode', 'lstm']:
    files = [file for file in os.listdir('plots/airplane_lat_long/' + model + '/')
             if (file.endswith('.csv') and not file.startswith('.'))]

    results_1 = []
    results_10 = []
    results_100 = []

    for file in files:
        path = 'plots/airplane_lat_long/' + model + '/' + file
        dataset = parse_filename(file)
        csv = pd.read_csv(path, header=0)
        if dataset == '1':
            results_1.append(csv.values[-1])
        if dataset == '10':
            results_10.append(csv.values[-1])
        if dataset == '100':
            results_100.append(csv.values[-1])
    n_1 = len(results_1)
    n_10 = len(results_10)
    n_100 = len(results_100)

    if n_1 > 0:
        max_1 = np.max(np.abs(results_1), axis=0)
    else:
        max_1 = np.nan
        results_1 = -np.ones((2, 12))
    if n_10 > 0:
        max_10 = np.max(np.abs(results_10), axis=0)
    else:
        max_10 = np.nan
        results_10 = -np.ones((2, 12))
    if n_100 > 0:
        max_100 = np.max(np.abs(results_100), axis=0)
    else:
        max_100 = np.nan
        results_100 = -np.ones((2, 12))

    results_1 = np.nanmean(np.abs(results_1), axis=0)
    results_10 = np.nanmean(np.abs(results_10), axis=0)
    results_100 = np.nanmean(np.abs(results_100), axis=0)

    np.set_printoptions(precision=5, suppress=True, linewidth=150)
    print('\n')
    print(model)
    print('--------------------------------')
    print('MAX')
    print('          ', [x[:12] for x in csv.columns.values])
    print('  1% (n={})'.format(n_1), max_1)
    print(' 10% (n={})'.format(n_10), max_10)
    print('100% (n={})'.format(n_100), max_100)

    print('AVERAGES')
    print('          ', [x[:12] for x in csv.columns.values])
    print('  1% (n={})'.format(n_1), results_1)
    print(' 10% (n={})'.format(n_10), results_10)
    print('100% (n={})'.format(n_100), results_100)
    print('--------------------------------')

    pretty_print_latex(results_1)
    pretty_print_latex(results_10)
    pretty_print_latex(results_100)

    if np.max(max_1/results_1) > 3:
        print('WARNING: 1% needs more trials')
    if np.max(max_10/results_10) > 3:
        print('WARNING: 10% needs more trials')
    if np.max(max_100/results_100) > 3:
        print('WARNING: 100% needs more trials')
    print('--------------------------------')
