import numpy as np
import tensorflow as tf

cx, cy, w, h = 0, 1, 2, 3


def anchor_IOU(boxes, anchors):
    boxes = tf.tile(tf.expand_dims(boxes, 1), [1, anchors.shape[0], 1])

    anchor_xmin = anchors[:, cx] - anchors[:, w] * 0.5
    anchor_ymin = anchors[:, cy] - anchors[:, h] * 0.5
    anchor_xmax = anchors[:, cx] + anchors[:, w] * 0.5
    anchor_ymax = anchors[:, cy] + anchors[:, h] * 0.5

    anchor_xmin = tf.tile(tf.expand_dims(anchor_xmin, 0), [boxes.shape[0], 1])
    anchor_ymin = tf.tile(tf.expand_dims(anchor_ymin, 0), [boxes.shape[0], 1])
    anchor_xmax = tf.tile(tf.expand_dims(anchor_xmax, 0), [boxes.shape[0], 1])
    anchor_ymax = tf.tile(tf.expand_dims(anchor_ymax, 0), [boxes.shape[0], 1])

    inter_w = tf.minimum(boxes[..., 2], anchor_xmax) - tf.maximum(boxes[..., 0], anchor_xmin)
    inter_h = tf.minimum(boxes[..., 3], anchor_ymax) - tf.maximum(boxes[..., 1], anchor_ymin)

    inter_w = tf.maximum(tf.cast(0, tf.float32), inter_w)
    inter_h = tf.maximum(tf.cast(0, tf.float32), inter_h)
    intersection = inter_w * inter_h

    area_boxes = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
    area_anchors = (anchor_xmax - anchor_xmin) * (anchor_ymax - anchor_ymin)
    union = area_boxes + area_anchors - intersection

    return tf.cast(intersection, tf.float32) / union


def encode(boxes, anchors, model,
           pos_iou_threshold=0.5, neg_iou_threshold=0.3):
    num_anchors = len(anchors)

    encoded = np.zeros((num_anchors, model.num_classes + 4),
                       dtype=np.float32)

    ious = anchor_IOU(boxes, anchors).numpy()

    # First find the best anchor for each gt
    def find_best_anchor_for_gts(ious):
        best_anchor_for_gt = {}
        for _ in range(len(boxes)):
            best_gt = tf.argmax(tf.reduce_max(ious, axis=1)).numpy()
            best_anchor = tf.argmax(ious[best_gt]).numpy()

            if ious[best_gt, best_anchor] >= neg_iou_threshold:
                best_anchor_for_gt[best_gt] = best_anchor
                ious[best_gt, :] = -1.0
                ious[:, best_anchor] = -1.0
        return best_anchor_for_gt

    best_anchor_for_gt = find_best_anchor_for_gts(ious.copy())

    if len(best_anchor_for_gt) > 0:
        ious[np.array(list(best_anchor_for_gt.keys())),
             np.array(list(best_anchor_for_gt.values()))] = 2.0

    best_iou_for_anchor = tf.reduce_max(ious, axis=0)
    best_gt_for_anchor = tf.argmax(ious, axis=0)

    pos_mask = (best_iou_for_anchor >= pos_iou_threshold).numpy()
    neg_mask = (best_iou_for_anchor < neg_iou_threshold).numpy()

    encoded[neg_mask, 0] = 1.0

    boxes_cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    boxes_cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    boxes_w = boxes[:, 2] - boxes[:, 0]
    boxes_h = boxes[:, 3] - boxes[:, 1]
    boxes_label = boxes[:, 4]

    matched_gt_idx = tf.boolean_mask(best_gt_for_anchor, mask=pos_mask)
    boxes_cx = tf.gather(boxes_cx, indices=matched_gt_idx, axis=0)
    boxes_cy = tf.gather(boxes_cy, indices=matched_gt_idx, axis=0)
    boxes_w = tf.gather(boxes_w, indices=matched_gt_idx, axis=0)
    boxes_h = tf.gather(boxes_h, indices=matched_gt_idx, axis=0)
    boxes_label = tf.gather(boxes_label, indices=matched_gt_idx, axis=0)

    pos_anchors = tf.boolean_mask(anchors, mask=pos_mask)
    anchor_cx = pos_anchors[:, cx]
    anchor_cy = pos_anchors[:, cy]
    anchor_w = pos_anchors[:, w]
    anchor_h = pos_anchors[:, h]

    encoded[pos_mask, tf.cast(boxes_label, tf.int32)] = 1.0
    encoded[pos_mask, model.num_classes+cx] = (boxes_cx - anchor_cx) / anchor_w
    encoded[pos_mask, model.num_classes+cy] = (boxes_cy - anchor_cy) / anchor_h
    encoded[pos_mask, model.num_classes+w] = tf.math.log(boxes_w / anchor_w)
    encoded[pos_mask, model.num_classes+h] = tf.math.log(boxes_h / anchor_h)
    encoded[pos_mask, model.num_classes:] /= model.variances

    return encoded


def nms(boxes, nms_threshold=0.5):
    from .metrics import IOU

    if len(boxes) == 0:
        return tf.zeros((0, 6), dtype=tf.float32)

    idxs = tf.argsort(boxes[:, 5], direction='DESCENDING')
    boxes = tf.gather_nd(boxes, indices=tf.expand_dims(idxs, axis=-1))

    output_boxes = []

    while len(boxes) > 0:
        box = boxes[0]
        output_boxes.append(box)

        boxes = boxes[1:]
        if len(boxes) == 0:
            break

        ious = IOU(box, boxes)
        suppress = ious >= nms_threshold
        boxes = tf.boolean_mask(boxes, mask=~suppress)

    return tf.stack(output_boxes, axis=0)


def decode(output, anchors, model, from_logits=True,
           conf_threshold=0.5, nms_threshold=0.5):
    cls, loc = output[..., :-4], output[..., -4:]

    if from_logits:
        cls = tf.nn.softmax(cls, axis=-1)

    boxes = []

    for class_id in range(1, model.num_classes):
        pos_mask = cls[:, class_id] >= conf_threshold

        if not tf.reduce_any(pos_mask):
            continue

        class_conf = tf.boolean_mask(cls[:, class_id], mask=pos_mask)
        class_loc = tf.boolean_mask(loc, mask=pos_mask)
        pos_anchors = tf.boolean_mask(anchors, mask=pos_mask)

        class_loc *= model.variances
        boxes_cx = class_loc[:, cx] * pos_anchors[:, w] + pos_anchors[:, cx]
        boxes_cy = class_loc[:, cy] * pos_anchors[:, h] + pos_anchors[:, cy]
        boxes_w = tf.math.exp(class_loc[:, w]) * pos_anchors[:, w]
        boxes_h = tf.math.exp(class_loc[:, h]) * pos_anchors[:, h]

        boxes_xmin = boxes_cx - boxes_w * 0.5
        boxes_ymin = boxes_cy - boxes_h * 0.5
        boxes_xmax = boxes_cx + boxes_w * 0.5
        boxes_ymax = boxes_cy + boxes_h * 0.5

        class_boxes = tf.stack([boxes_xmin, boxes_ymin, boxes_xmax, boxes_ymax,
                                tf.zeros_like(class_conf)+class_id, class_conf], axis=-1)
        boxes.append(nms(class_boxes, nms_threshold))

    if len(boxes) == 0:
        boxes = tf.zeros((0, 6), dtype=tf.float32)
    else:
        boxes = tf.concat(boxes, axis=0)

    return boxes
