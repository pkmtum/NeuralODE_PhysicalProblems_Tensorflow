import pandas as pd
import numpy as np
import os

def parse_filename(file):
    name = file.split('results')[-1]
    if '100' in name:
        return '100'
    if '10' in name:
        return '10'
    if '1' in name:
        return '1'


for model in ['odenet', 'densenet', 'learnedode', 'lstm']:
    files = [file for file in os.listdir('plots/airplane/' + model + '/') if (file.endswith('.csv') and not file.startswith('.'))]

    results_1 = []
    results_10 = []
    results_100 = []

    for file in files:
        path = 'plots/airplane/' + model + '/' + file
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

    if n_1 != 0:
        max_1 = np.max(np.abs(results_1), axis=0)
    else:
        max_1 = np.nan
    if n_10 != 0:
        max_10 = np.max(np.abs(results_10), axis=0)
    else:
        max_10 = np.nan
    if n_100 != 0:
        max_100 = np.max(np.abs(results_100), axis=0)
    else:
        max_100 = np.nan

    results_1 = np.mean(np.abs(results_1), axis=0)
    results_10 = np.mean(np.abs(results_10), axis=0)
    results_100 = np.mean(np.abs(results_100), axis=0)

    np.set_printoptions(precision=6, suppress=True, linewidth=150)
    print(model, 'MAX')
    print('          ', [x[:8] for x in csv.columns.values])
    print('  1% (n={})'.format(n_1), max_1)
    print(' 10% (n={})'.format(n_10), max_10)
    print('100% (n={})'.format(n_100), max_100)

    print(model, 'AVERAGES')
    print('          ', [x[:8] for x in csv.columns.values])
    print('  1% (n={})'.format(n_1), results_1)
    print(' 10% (n={})'.format(n_10), results_10)
    print('100% (n={})'.format(n_100), results_100)

    if np.max(max_1/results_1) > 3:
        print('WARNING: 1% needs more trials')
    if np.max(max_10/results_10) > 3:
        print('WARNING: 10% needs more trials')
    if np.max(max_100/results_100) > 3:
        print('WARNING: 100% needs more trials')
