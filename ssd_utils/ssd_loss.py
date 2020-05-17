import tensorflow as tf


class SSDLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, from_logits=True, **kwargs):
        self.gamma = gamma
        self.from_logits = from_logits
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        true_cls, true_loc = y_true[..., :-4], y_true[..., -4:]
        pred_cls, pred_loc = y_pred[..., :-4], y_pred[..., -4:]

        pos_mask = tf.reduce_sum(true_cls[..., 1:], axis=-1)
        N = tf.reduce_sum(pos_mask)
        N = tf.maximum(tf.ones_like(N), N)
        pos_mask = tf.cast(pos_mask, tf.bool)

        cls_loss = tf.reduce_sum(
            self.focal_loss(true_cls, pred_cls)) / N

        true_loc = tf.boolean_mask(true_loc, mask=pos_mask)
        pred_loc = tf.boolean_mask(pred_loc, mask=pos_mask)
        loc_loss = tf.reduce_sum(
            self.smooth_l1_loss(true_loc, pred_loc)) / N

        return cls_loss + loc_loss

    def focal_loss(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        cross_entropy = - y_true * tf.math.log(y_pred + 1e-8)
        loss = tf.math.pow(1.0 - y_pred, self.gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    def smooth_l1_loss(self, y_true, y_pred):
        x = tf.abs(y_true - y_pred)
        return tf.where(x < 1.0, 0.5*x*x, x - 0.5)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'gamma': self.gamma,
                'from_logits': self.from_logits}
