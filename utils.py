import numpy as np
import os
import sys
import tensorflow as tf

from ssd_utils import output_encoder
from ssd_utils.metrics import MeanAveragePrecision

def import_by_name(module):
    comps = module.split('.')
    mod = __import__(comps[0])
    for comp in comps[1:]:
        mod = getattr(mod, comp)
    return mod

def train_test_split_tensors(*arrays, test_size, random_state=None):
    num_samples = len(arrays[0])
    
    np.random.seed(random_state)
    perm = np.random.permutation(num_samples)
    np.random.seed(None)

    train_idxs = perm[test_size:]
    test_idxs = perm[:test_size]

    splits = []
    for arr in arrays:
        splits.append(tf.gather(arr, indices=train_idxs, axis=0))
        splits.append(tf.gather(arr, indices=test_idxs, axis=0))

    return splits

class MeanAveragePrecisionCallback(tf.keras.callbacks.Callback):
    def __init__(self, data, anchors, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data
        self.anchors = anchors
        self.mean_AP_metric = MeanAveragePrecision()

    def on_epoch_end(self, epoch, logs=None):
        for x, y_true in self.data:
            predictions = [output_encoder.decode(y, self.anchors, self.model)
                           for y in self.model(x)]
            ground_truth = [y.to_tensor() for y in y_true]
            self.mean_AP_metric.update_state(ground_truth, predictions)

        logs['val_meanAP'] = self.mean_AP_metric.result().numpy()
        self.mean_AP_metric.reset_state()
