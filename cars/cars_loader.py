import numpy as np
import os
from os import path
import pandas as pd
import sys
import tensorflow as tf

root_path = os.path.abspath(os.path.join('..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from ssd_utils import output_encoder
import config

def read_and_resize_image(image_path, boxes, new_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    if new_size is not None:
        prior_shape = tf.shape(image)
        boxes = tf.cast(boxes, tf.float32)
        
        xmin = boxes[:, 0] * tf.cast(new_size[1] / prior_shape[1], tf.float32)
        ymin = boxes[:, 1] * tf.cast(new_size[0] / prior_shape[0], tf.float32)
        xmax = boxes[:, 2] * tf.cast(new_size[1] / prior_shape[1], tf.float32)
        ymax = boxes[:, 3] * tf.cast(new_size[0] / prior_shape[0], tf.float32)
        labels = tf.cast(boxes[:, 4], xmin.dtype)

        boxes = tf.stack([xmin, ymin, xmax, ymax, labels], axis=-1)
        image = tf.image.resize(image, new_size)
        image = tf.cast(image, dtype=tf.uint8)
        boxes = tf.cast(boxes, dtype=tf.float32)
        boxes = tf.RaggedTensor.from_tensor(boxes)

    return image, boxes


def load_fake_data(root):
    annotations_file = path.join(root, "cars_fake.csv")
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

    image_paths = tf.convert_to_tensor(image_paths)
    xmins = tf.ragged.constant(xmins, dtype='int32')
    ymins = tf.ragged.constant(ymins, dtype='int32')
    xmaxs = tf.ragged.constant(xmaxs, dtype='int32')
    ymaxs = tf.ragged.constant(ymaxs, dtype='int32')
    labels = tf.ones_like(xmins, dtype='int32')
    bnd_boxes = tf.stack([xmins, ymins, xmaxs, ymaxs, labels], axis=2)

    print('Loaded {} image samples'.format(len(image_paths)))
    return image_paths, bnd_boxes

                    

def load_cars(root=path.join("..", "data", "stanford_cars"),
                  split='train'):
    if split == 'fake':
        return load_fake_data(root)

    mode = 'test' if split == 'test' else 'train'
    data = pd.read_csv(os.path.join(root, 'devkit', 'cars_%s_annos.csv'%mode))

    if split == 'train':
        data = data.iloc[config.VALID_SIZE:]
    elif split == 'valid':
        data = data.iloc[:config.VALID_SIZE]

    image_paths = []
    xmins, ymins = [], []
    xmaxs, ymaxs = [], []

    for row in data.itertuples(index=False):
        image_file, xmin, ymin, xmax, ymax = row
        image_path = os.path.join(root, 'cars_%s'%mode, image_file)

        image_paths.append(image_path)
        xmins.append([xmin])
        ymins.append([ymin])
        xmaxs.append([xmax])
        ymaxs.append([ymax])

    image_paths = tf.convert_to_tensor(image_paths)
    xmins = tf.ragged.constant(xmins, dtype='float32')
    ymins = tf.ragged.constant(ymins, dtype='float32')
    xmaxs = tf.ragged.constant(xmaxs, dtype='float32')
    ymaxs = tf.ragged.constant(ymaxs, dtype='float32')
    labels = tf.ones_like(xmins, dtype='float32')
    bnd_boxes = tf.stack([xmins, ymins, xmaxs, ymaxs, labels], axis=2)

    print('Loaded {} image samples'.format(len(image_paths)))
    return image_paths, bnd_boxes


def build_dataset(image_paths, bnd_boxes, image_size=None, batch_size=None,
                  repeat=False, shuffle=False, shuffle_buffer=100,
                  encode_output=False, anchors=None, model=None):

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, bnd_boxes))

    if repeat:
        dataset = dataset.repeat()

    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer,
                                  reshuffle_each_iteration=True)

    if image_size is not None:
        dataset = dataset.map(lambda x, y:
                              read_and_resize_image(x, y, new_size=image_size))

    if encode_output:
        def encode_fn(image, boxes):
            encoded = output_encoder.encode(boxes,
                                            anchors=anchors,
                                            model=model)
            return image, encoded
        dataset = dataset.map(lambda x, y:
                              tf.py_function(encode_fn,
                                             [x, y.to_tensor()],
                                             [tf.uint8, tf.float32]))
        input_shape = image_size + (3,)
        output_shape = anchors.shape[:1] + (4 + model.num_classes,)
        dataset = dataset.map(lambda x, y: (tf.reshape(x, input_shape),
                                            tf.reshape(y, output_shape)))

    if batch_size is not None:
        dataset = dataset.batch(batch_size, drop_remainder=False)

    return dataset
