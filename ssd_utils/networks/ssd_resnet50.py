import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import resnet50
from tensorflow.keras import layers


class SSDResNet50(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 scales=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
                 base_layers=['conv4_block4_out', 'conv4_block5_out',
                              'conv4_block6_out', 'conv5_block1_out',
                              'conv5_block2_out', 'conv5_block3_out'],
                 aspect_ratios=[0.5, 1.0, 2.0],
                 variances=[0.1, 0.1, 0.2, 0.2],
                 weight_decay=5e-4,
                 **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes

        if len(scales) != len(base_layers):
            raise Exception('You need to provide one scale for each base layer.')
        self.scales = scales
        self.base_layers = base_layers

        self.aspect_ratios = aspect_ratios
        self.boxes_per_cell = len(aspect_ratios)

        if len(variances) != 4:
            raise Exception('You need to provide exactly 4 variance values \
            (one for each bounding box parameter).')
        self.variances = variances

        backbone = resnet50.ResNet50(include_top=False, weights='imagenet')

        self.get_base_features = tf.keras.Model(
            inputs=backbone.layers[0].input,
            outputs=[
                backbone.get_layer(layer_name).output
                for layer_name in self.base_layers
            ])

        self.conv_cls = []
        self.conv_loc = []

        for idx in range(len(self.scales)):
            self.conv_cls.append(
                layers.Conv2D(
                    filters=self.boxes_per_cell * self.num_classes,
                    kernel_size=3,
                    padding='same',
                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                    name='conv_cls_{}'.format(idx+1)
                )
            )
            self.conv_loc.append(
                layers.Conv2D(
                    filters=self.boxes_per_cell * 4,
                    kernel_size=3,
                    padding='same',
                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                    name='conv_loc_{}'.format(idx+1)
                )
            )

    def call(self, inputs, training=False):
        inputs = tf.convert_to_tensor(tf.cast(inputs, tf.float32))
        inputs = resnet50.preprocess_input(inputs)
        base_feature_maps = self.get_base_features(inputs, training=training)

        if len(self.scales) == 1:
            base_feature_maps = [base_feature_maps]

        self.feature_shapes = []
        cls_output, loc_output = [], []

        for idx in range(len(self.scales)):
            cls = self.conv_cls[idx](base_feature_maps[idx])
            loc = self.conv_loc[idx](base_feature_maps[idx])
            self.feature_shapes.append((cls.shape[1], cls.shape[2]))
            cls = layers.Reshape((-1, self.num_classes))(cls)
            loc = layers.Reshape((-1, 4))(loc)
            cls_output.append(cls)
            loc_output.append(loc)

        if len(self.scales) == 1:
            cls_output = cls_output[0]
            loc_output = loc_output[0]
        else:
            cls_output = layers.Concatenate(axis=1)(cls_output)
            loc_output = layers.Concatenate(axis=1)(loc_output)

        return layers.concatenate([cls_output, loc_output], axis=-1)

    def get_anchors(self, image_shape, build=True):
        if build:
            self.build(
                input_shape=(None, image_shape[0], image_shape[1], image_shape[2]))

        input_height, input_width = image_shape[:2]
        anchors = []

        for scale, (feature_height, feature_width) in zip(self.scales,
                                                          self.feature_shapes):
            center_y, center_x = tf.meshgrid(tf.range(feature_height),
                                             tf.range(feature_width),
                                             indexing='ij')
            center_x = tf.cast(center_x, tf.float32)
            center_y = tf.cast(center_y, tf.float32)
            center_x = (center_x + 0.5) * input_width / feature_width
            center_y = (center_y + 0.5) * input_height / feature_height

            center_x = tf.expand_dims(center_x, axis=-1)
            center_y = tf.expand_dims(center_y, axis=-1)
            center_x = tf.tile(center_x, multiples=[1, 1, self.boxes_per_cell])
            center_y = tf.tile(center_y, multiples=[1, 1, self.boxes_per_cell])

            width = np.zeros(shape=self.boxes_per_cell, dtype=np.float32)
            height = np.zeros(shape=self.boxes_per_cell, dtype=np.float32)

            for i in range(self.boxes_per_cell):
                width[i] = input_width * scale * np.sqrt(self.aspect_ratios[i])
                height[i] = input_height * scale / np.sqrt(self.aspect_ratios[i])

            width = tf.clip_by_value(width, 0, input_width)
            height = tf.clip_by_value(height, 0, input_height)

            width = tf.zeros_like(center_x) + width
            height = tf.zeros_like(center_x) + height

            feature_anchors = tf.stack([center_x, center_y, width, height], axis=-1)
            feature_anchors = tf.reshape(feature_anchors, shape=(-1, 4))
            anchors.append(feature_anchors)

        return tf.concat(anchors, axis=0)
