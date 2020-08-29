import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

TOTAL_SAMPLES = config.TRAIN_SAMPLES[-1]

results = pd.read_csv('results.csv')
groups = results.groupby(['architecture', 'train_type']).groups

for architecture in config.ARCHITECTURES:
    print("Results for {}".format(architecture))
    plt.figure(figsize=(9, 7))
    
    lines = results.iloc[groups[(architecture, 'pretrained')]]

    meanAPs = lines[["run_{}".format(run)
                   for run in range(1, config.NUM_RUNS+1)]]

    avg_meanAP = np.array(meanAPs).mean()*100
    std_meanAP = np.array(meanAPs).std()*100

    #print("\n{} pretrained only: {:.2f}% +- {:.2f}%".format(architecture, avg_meanAP, std_meanAP))
    print("\\multirow{%d}{*}{Birds} " % (len(config.TRAIN_SAMPLES)+1), end="")
    print("& pretrain only(0\\%c)   &  --  & %.02lf\\%c $\\pm$ %.02lf\\%c  \\\\ \\cline{2-4}" % (
        '%', avg_meanAP, '%', std_meanAP, '%'))

    for num_samples in config.TRAIN_SAMPLES:
        print("& %d ($\\sim$%.0lf\\%c)" % (
            num_samples, 100 * num_samples / TOTAL_SAMPLES, '%'), end="")

        for train_type in ['from_scratch', 'finetuned']:
            lines = results.iloc[groups[(architecture, train_type)]]

            train_samples = lines["train_samples"]
            meanAPs = lines[train_samples == num_samples][["run_{}".format(run)
                             for run in range(1, config.NUM_RUNS+1)]]

            avg_meanAP = float(meanAPs.mean(axis=1)) * 100
            std_meanAP = float(meanAPs.std(axis=1)) * 100

            #print(avg_meanAP, std_meanAP)
            print("   & %.2lf\\%c $\\pm$ %.2lf\\%c" % (avg_meanAP, '%', std_meanAP, '%'), end="")

        print("   \\\\ \\cline{2-4}")

    print("\n\n")

