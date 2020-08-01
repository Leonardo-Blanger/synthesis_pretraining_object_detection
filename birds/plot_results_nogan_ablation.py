import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config_nogan_ablation as config

results = pd.read_csv('results_nogan_ablation.csv')
groups = results.groupby(['architecture']).groups

for architecture in config.ARCHITECTURES:
    print('Results for {}'.format(architecture))
    
    
    lines = results.iloc[groups[architecture]]
    meanAPs = lines[['run_{}'.format(run) for run in range(1, config.NUM_RUNS+1)]]

    avg_meanAP = np.array(meanAPs).mean()*100
    std_meanAP = np.array(meanAPs).std()*100

    print('\t{:.2f} +- {:.2f}'.format(avg_meanAP, std_meanAP))
    print()
    print('-----------\n')
