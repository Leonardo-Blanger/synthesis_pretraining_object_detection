import numpy as np
import tensorflow as tf

from networks.ssd_mobilenet import SSDMobileNet
from output_encoder import encode, decode
#from metrics import ssd_losses

model = SSDMobileNet(num_classes=2,
                     scales=[0.3],
                     base_layers=['conv_pw_11_relu'],
                     aspect_ratios=[1.0])

image_shape = (480,480,3)
anchors = model.get_anchors(image_shape)

images = np.ones((1,) + image_shape)
boxes_batch = [np.array([[10, 10, 150, 150, 1]], dtype=np.float32)]

cls, loc = encode(boxes_batch, anchors, model)

print(cls.shape, loc.shape)
print(tf.reduce_sum(cls[..., 0]), tf.reduce_sum(cls[..., 1]))

boxes_decoded = decode((cls, loc), anchors, model, from_logits=False)

print(len(boxes_decoded))
print(boxes_decoded)
