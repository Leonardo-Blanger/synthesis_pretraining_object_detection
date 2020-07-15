import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config_pretraining_ablation as config

results = pd.read_csv('results_pretraining_ablation.csv')
groups = results.groupby(['architecture', 'train_type', 'num_fake_samples']).groups

for architecture in config.ARCHITECTURES:
    print('Results for {}'.format(architecture))
    
    for train_type in ['mixed', 'finetuned']:
        for prop_fake_samples in config.PROP_FAKE_SAMPLES:
            num_fake_samples = int(config.NUM_REAL_SAMPLES * prop_fake_samples)
            lines = results.iloc[groups[(architecture, train_type, num_fake_samples)]]
            meanAPs = lines[['run_{}'.format(run) for run in range(1, config.NUM_RUNS+1)]]

            avg_meanAP = np.array(meanAPs).mean()*100
            std_meanAP = np.array(meanAPs).std()*100

            print('\t{} {}x : {:.2f} +- {:.2f}'.format(train_type, prop_fake_samples,
                                                       avg_meanAP, std_meanAP))
        print()
    print('-----------\n')
