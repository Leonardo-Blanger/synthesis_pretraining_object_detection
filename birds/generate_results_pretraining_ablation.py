import pandas as pd
import os
import sys
import tensorflow as tf

root_path = os.path.abspath('..')
if root_path not in sys.path:
    sys.path.append(root_path)


import config_pretraining_ablation as config
from birds_loader import load_birds, build_dataset
from ssd_utils import output_encoder
from ssd_utils.metrics import MeanAveragePrecision
from utils import import_by_name, MeanAveragePrecisionCallback

test_image_paths, test_bnd_boxes = load_birds(split='test')

test_data = build_dataset(test_image_paths, test_bnd_boxes,
                          image_size=config.IMAGE_SIZE,
                          batch_size=config.BATCH_SIZE)

meanAP_metric = MeanAveragePrecision()
results = {'architecture': [],
           'num_real_samples': [],
           'num_fake_samples': [],
           'train_type': []}

for run in range(1, config.NUM_RUNS+1):
    results['run_{}'.format(run)] = []

for architecture in config.ARCHITECTURES:
    model_class = import_by_name('ssd_utils.networks.' + architecture)
    model = model_class(num_classes=len(config.CLASSES))
    anchors = model.get_anchors(image_shape=config.IMAGE_SIZE + (3,))

    for train_type in ['mixed', 'finetuned']:
        for prop_fake_samples in config.PROP_FAKE_SAMPLES:
            num_real_samples = config.NUM_REAL_SAMPLES
            num_fake_samples = int(prop_fake_samples * num_real_samples)
            
            results['architecture'].append(architecture)
            results['num_real_samples'].append(num_real_samples)
            results['num_fake_samples'].append(num_fake_samples)
            results['train_type'].append(train_type)

            for run in range(1, config.NUM_RUNS+1):
                weights_dir = 'weights_pretraining_ablation_{}'.format(run)

                model_name = '{}_{}_{}x'.format(architecture.lower(), train_type,
                                                prop_fake_samples)
                print('\nGenerating results for {}'.format(model_name))

                model_file = model_name + '.h5'
                model_path = os.path.join(weights_dir, model_file)

                model.load_weights(model_path)
                meanAP_metric.reset_state()

                for x, y_true in test_data:
                    ground_truth = [y.to_tensor() for y in y_true]
                    predictions = [output_encoder.decode(y, anchors, model)
                                   for y in model(x)]
                    meanAP_metric.update_state(ground_truth, predictions)

                test_meanAP = meanAP_metric.result().numpy()
                results['run_{}'.format(run)].append(test_meanAP)
                print('test meanAP:', test_meanAP)
    del model

results = pd.DataFrame(results)
results.to_csv('results_pretraining_ablation.csv', index=False)
