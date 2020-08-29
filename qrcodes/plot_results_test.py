import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

results = pd.read_csv('results.csv')
groups = results.groupby(['architecture', 'train_type']).groups

for architecture in config.ARCHITECTURES:
    plt.figure(figsize=(9, 7))
    
    lines = results.iloc[groups[(architecture, 'pretrained')]]

    meanAPs = lines[["run_{}".format(run)
                   for run in range(1, config.NUM_RUNS+1)]]

    avg_meanAP = np.array(meanAPs).mean()*100
    std_meanAP = np.array(meanAPs).std()*100

    plt.hlines(y=avg_meanAP,
               xmin=-config.TRAIN_SAMPLES[-1],
               xmax=2*config.TRAIN_SAMPLES[-1],
               linestyles='dashed')

    print("\n{} pretrained only: {:.2f}% +- {:.2f}%".format(architecture, avg_meanAP, std_meanAP))

    xs = np.arange(-config.TRAIN_SAMPLES[-1], 2*config.TRAIN_SAMPLES[-1], 0.01)
    y_min = np.zeros_like(xs) + avg_meanAP - std_meanAP
    y_max = np.zeros_like(xs) + avg_meanAP + std_meanAP

    plt.fill_between(xs, y_min, y_max, color='0.8')

    for train_type in ['from_scratch', 'finetuned']:
        lines = results.iloc[groups[(architecture, train_type)]]

        train_samples = lines["train_samples"]
        meanAPs = lines[["run_{}".format(run)
                         for run in range(1, config.NUM_RUNS+1)]]

        avg_meanAP = meanAPs.mean(axis=1)*100
        std_meanAP = meanAPs.std(axis=1)*100

        for num_samples, mAP, std in zip(config.TRAIN_SAMPLES, avg_meanAP, std_meanAP):
            print("\n{} {} on {} samples: {:.2f}% +- {:.2f}%".format(architecture, train_type, num_samples, mAP, std))

        plt.errorbar(train_samples, avg_meanAP, yerr=std_meanAP, label=train_type)


    Xs, X_TICKS = [], []
    ALL_SAMPLES = config.TRAIN_SAMPLES[-1]
    for num_samples in config.TRAIN_SAMPLES:
        #if num_samples == 10 or num_samples%100 == 0:
        Xs.append(num_samples)
        X_TICKS.append('{}\n~{:.0f}%'.format(num_samples, 100*num_samples/ALL_SAMPLES))

    plt.xticks(Xs, X_TICKS)
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.xlim(xmin = 0, xmax = config.TRAIN_SAMPLES[-1]+10)
    plt.ylim(ymax = 75)
    plt.title(architecture + " (QR Codes)", size=20)
    plt.xlabel("# Train Samples", size=18)
    plt.ylabel("Test meanAP@0.5 (%)", size=18)
    plt.legend(loc="lower right", fontsize=17)
    plt.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.101)
    plt.tight_layout()
    
    plt.savefig(architecture.lower() + '_test_results_qrcodes.png')
    plt.show()
