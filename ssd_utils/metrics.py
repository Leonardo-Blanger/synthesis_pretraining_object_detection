import numpy as np
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm


def IOU(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = [box1[..., i] for i in range(4)]
    xmin2, ymin2, xmax2, ymax2 = [box2[..., i] for i in range(4)]

    inter_width = tf.minimum(xmax1, xmax2) - tf.maximum(xmin1, xmin2)
    inter_height = tf.minimum(ymax1, ymax2) - tf.maximum(ymin1, ymin2)
    inter_width = tf.maximum(0, inter_width)
    inter_height = tf.maximum(0, inter_height)
    intersection = inter_width * inter_height

    sum_areas = (xmax1-xmin1)*(ymax1-ymin1) + (xmax2-xmin2)*(ymax2-ymin2)
    union = sum_areas - intersection

    return intersection / union


def AP(batch_ground_truth, batch_predictions, iou_threshold=0.5):
    all_predictions = []
    total_positives = 0

    for ground_truth, predictions in tqdm(zip(batch_ground_truth, batch_predictions),
                                          desc='Calculating Average Precision...'):
        total_positives += len(ground_truth)

        predictions = sorted(predictions, key=lambda x: x[5].numpy(), reverse=True)
        matched = np.zeros(len(ground_truth))

        for pred in predictions:
            if len(ground_truth) == 0:
                all_predictions.append((pred[5].numpy(), False))
                continue

            iou = IOU(pred, ground_truth)
            i = tf.argmax(iou)

            if iou[i].numpy() >= iou_threshold and not matched[i]:
                all_predictions.append((pred[5].numpy(), True))
                matched[i] = True
            else:
                all_predictions.append((pred[5].numpy(), False))

    all_predictions = sorted(all_predictions, reverse=True)

    recalls, precisions = [0], [1]
    TP, FP = 0, 0

    for _, result in all_predictions:
        if result:
            TP += 1
        else:
            FP += 1
        precisions.append(TP / (TP+FP))
        recalls.append(TP / total_positives)

    recalls = np.array(recalls)
    precisions = np.array(precisions)

    for i in range(len(precisions)-2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])

    return np.sum((recalls[1:]-recalls[:-1]) * precisions[1:])


def per_class_AP(ground_truth, predictions, iou_threshold=0.5):
    classes = np.unique(
        np.concatenate([np.unique(boxes[:, 4])
                        for boxes in ground_truth]))
    APs = []

    for cls in classes:
        print('\nCalculating Average Precision for class', cls)

        class_ground_truth = [
            tf.boolean_mask(boxes, mask=tf.equal(boxes[:, 4], cls))
            for boxes in ground_truth
        ]
        class_predictions = [
            tf.boolean_mask(boxes, mask=tf.equal(boxes[:, 4], cls))
            for boxes in predictions
        ]

        APs.append(AP(class_ground_truth, class_predictions, iou_threshold))

    return APs


def mean_AP(ground_truth, y_pred, iou_threshold=0.5):
    return np.mean(per_class_AP(ground_truth, y_pred, iou_threshold))
