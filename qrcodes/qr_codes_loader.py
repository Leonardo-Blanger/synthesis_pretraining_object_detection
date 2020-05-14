import numpy as np
from os import path
import pandas as pd
import tensorflow as tf


def load_qr_codes_dataset(root=path.join("data", "qr_codes"),
                          split='train'):
    annotations_file = path.join(root, "qr_codes_{}.csv".format(split))
    annotations = pd.read_csv(annotations_file,
                              dtype={'image_id': str})
    annotations = annotations.groupby('image_id')

    image_paths = []
    xmins, ymins = [], []
    xmaxs, ymaxs = [], []

    for image_id, bnd_boxes in annotations:
        image_path = path.join(root, 'images', image_id+'.jpg')
        xmin, ymin, xmax, ymax = [bnd_boxes[field].to_numpy(dtype='int32')
                                  for field in ('xmin', 'ymin', 'xmax', 'ymax')]

        image_paths.append(image_path)
        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)

    xmins = tf.ragged.constant(xmins)
    ymins = tf.ragged.constant(ymins)
    xmaxs = tf.ragged.constant(xmaxs)
    ymaxs = tf.ragged.constant(ymaxs)
    bnd_boxes = tf.stack([xmins, ymins, xmaxs, ymaxs], axis=2)

    return tf.data.Dataset.from_tensor_slices((image_paths, bnd_boxes))
