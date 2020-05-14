from os import path
import pandas as pd
import tensorflow as tf


def load_image(image_path, boxes, new_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    prior_shape = tf.shape(image)

    xmin = boxes[:, 0] * new_size[1] / prior_shape[1]
    ymin = boxes[:, 1] * new_size[0] / prior_shape[0]
    xmax = boxes[:, 2] * new_size[1] / prior_shape[1]
    ymax = boxes[:, 3] * new_size[0] / prior_shape[0]

    boxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
    image = tf.image.resize(image, new_size)
    image = tf.cast(image, dtype=tf.uint8)

    return image, boxes


def load_qr_codes_dataset(root=path.join("..", "data", "qr_codes"),
                          split='train',
                          image_size=None):
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

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, bnd_boxes))

    if image_size is not None:
        dataset = dataset.map(lambda x, y: load_image(x, y, image_size))

    return dataset
