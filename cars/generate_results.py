import pandas as pd
import os
import sys
import tensorflow as tf

root_path = os.path.abspath('..')
if root_path not in sys.path:
    sys.path.append(root_path)


import config
from cars_loader import load_cars, build_dataset
from ssd_utils import output_encoder
from ssd_utils.metrics import MeanAveragePrecision
from utils import import_by_name, MeanAveragePrecisionCallback

test_image_paths, test_bnd_boxes = load_cars(split='test')

test_data = build_dataset(test_image_paths, test_bnd_boxes,
                          image_size=config.IMAGE_SIZE,
                          batch_size=config.BATCH_SIZE)

meanAP_metric = MeanAveragePrecision()
results = {'architecture': [],
           'train_samples': [],
           'train_type': []}

for run in range(1, config.NUM_RUNS+1):
    results['run_{}'.format(run)] = []

for architecture in config.ARCHITECTURES:
    model_class = import_by_name('ssd_utils.networks.' + architecture)
    model = model_class(num_classes=len(config.CLASSES))
    anchors = model.get_anchors(image_shape=config.IMAGE_SIZE + (3,))

    train_type = 'pretrained'
    results['architecture'].append(architecture)
    results['train_samples'].append(None)
    results['train_type'].append(train_type)

    for run in range(1, config.NUM_RUNS+1):
        weights_dir = 'weights_{}'.format(run)

        model_name = architecture.lower() + '_{}'.format(train_type)
        print('\nGenerating results for {}'.format(model_name))

        model_file = model_name + '.h5'
        model_path = os.path.join(weights_dir, model_file)

        if not os.path.exists(model_path):
            raise Exception('Model weights at {} not found'.format(model_path))

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

    for train_samples in config.TRAIN_SAMPLES:
        for train_type in ['from_scratch', 'finetuned']:

            results['architecture'].append(architecture)
            results['train_samples'].append(train_samples)
            results['train_type'].append(train_type)
                
            for run in range(1, config.NUM_RUNS+1):
                weights_dir = 'weights_{}'.format(run)

                model_name = architecture.lower() + '_{}_samples_{}'.format(
                    train_samples, train_type)
                print('\nGenerating results for {}'.format(model_name))
                
                model_file = model_name + '.h5'
                model_path = os.path.join(weights_dir, model_file)

                if not os.path.exists(model_path):
                    raise Exception('Model weights at {} not found'.format(model_path))

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
results.to_csv('results.csv', index=False)
