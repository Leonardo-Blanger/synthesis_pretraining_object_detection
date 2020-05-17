from os import path
import pandas as pd
import tensorflow as tf


def read_and_resize_image(image_path, boxes, new_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)

    if new_size is not None:
        prior_shape = tf.shape(image)

        xmin = boxes[:, 0] * new_size[1] / prior_shape[1]
        ymin = boxes[:, 1] * new_size[0] / prior_shape[0]
        xmax = boxes[:, 2] * new_size[1] / prior_shape[1]
        ymax = boxes[:, 3] * new_size[0] / prior_shape[0]
        labels = tf.cast(boxes[:, 4], xmin.dtype)

        boxes = tf.stack([xmin, ymin, xmax, ymax, labels], axis=-1)
        image = tf.image.resize(image, new_size)
        image = tf.cast(image, dtype=tf.uint8)
        boxes = tf.cast(boxes, dtype=tf.float32)
        boxes = tf.RaggedTensor.from_tensor(boxes)

    return image, boxes


def load_qr_codes_dataset(root=path.join("..", "data", "qr_codes"),
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

    xmins = tf.ragged.constant(xmins, dtype='int32')
    ymins = tf.ragged.constant(ymins, dtype='int32')
    xmaxs = tf.ragged.constant(xmaxs, dtype='int32')
    ymaxs = tf.ragged.constant(ymaxs, dtype='int32')
    labels = tf.ones_like(xmins, dtype='int32')
    bnd_boxes = tf.stack([xmins, ymins, xmaxs, ymaxs, labels], axis=2)

    return tf.data.Dataset.from_tensor_slices((image_paths, bnd_boxes))
