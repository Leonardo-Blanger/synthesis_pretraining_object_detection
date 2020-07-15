import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

import config

ALL_SAMPLES = config.TRAIN_SAMPLES[-1]

for architecture in config.ARCHITECTURES:
    plt.figure(figsize=(9, 7))

    for train_type in ['from_scratch', 'finetuned']:
        meanAPs = []
        
        for run in range(1, config.NUM_RUNS+1):
            history_file = os.path.join('history_{}'.format(run),
                            '{}_{}_samples_{}_history.pickle'.format(
                                architecture.lower(), ALL_SAMPLES, train_type))
            with open(history_file, 'rb') as f:
                history = pickle.load(f)
            val_meanAP = history['val_meanAP']
            
            meanAPs.append(val_meanAP)

        meanAPs = np.array(meanAPs)
        avg_meanAP = meanAPs.mean(axis=0) * 100
        std_meanAP = meanAPs.std(axis=0) * 100

        train_iterations = np.arange(1, config.NUM_EPOCHS+1) * config.STEPS_PER_EPOCH
        
        plt.errorbar(train_iterations, avg_meanAP, yerr=std_meanAP, label=train_type)
        plt.xticks(train_iterations[::2])

    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.ylim(ymax = 75)
    plt.title(architecture + " (QR Codes)", size=20)
    plt.xlabel("Train Iteration", size=16)
    plt.ylabel("Validation meanAP@0.5 (%)", size=16)
    plt.legend(loc="lower right")
    plt.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.08)
    plt.tight_layout()
    
    plt.savefig(architecture.lower() + '_valid_results_qrcodes.png')
    plt.show()
