from bs4 import BeautifulSoup
import imgaug as ia
import numpy as np
import os

def parse(xml_path):
    if not os.path.exists(xml_path):
        raise Exception('Annotation file %s not found' % xml_path)

    with open(xml_path, 'r') as f:
        annotation = BeautifulSoup(f, 'lxml')

    img_height = int(annotation.size.height.text)
    img_width = int(annotation.size.width.text)
    img_depth = int(annotation.size.depth.text)

    bnd_boxes = []

    for obj in annotation.find_all('object'):
        label = obj.find('name').text
        xmin = float(obj.bndbox.xmin.text) - 1
        ymin = float(obj.bndbox.ymin.text) - 1
        xmax = float(obj.bndbox.xmax.text) - 1
        ymax = float(obj.bndbox.ymax.text) - 1

        bnd_boxes.append(
            ia.BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax, label=label)
        )

    return ia.BoundingBoxesOnImage(bnd_boxes, shape=(img_height, img_width, img_depth))
