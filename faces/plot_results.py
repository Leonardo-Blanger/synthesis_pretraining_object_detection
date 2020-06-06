import matplotlib.pyplot as plt
import pandas as pd

import config

results = pd.read_csv('results.csv')
groups = results.groupby(['architecture', 'train_type']).groups

for architecture in config.ARCHITECTURES:
    for train_type in ['from_scratch', 'finetuned']:
        lines = results.iloc[groups[(architecture, train_type)]]

        train_samples = lines["train_samples"]
        meanAPs = lines[["run_{}".format(run)
                         for run in range(1, config.NUM_RUNS+1)]]

        avg_meanAP = meanAPs.mean(axis=1)*100
        std_meanAP = meanAPs.std(axis=1)*100
        
        plt.errorbar(train_samples, avg_meanAP, yerr=std_meanAP, label=train_type)

    plt.title(architecture)
    plt.xlabel("# Train Samples")
    plt.ylabel("Test meanAP (%)")
    plt.legend(loc="lower right")
    plt.show()
