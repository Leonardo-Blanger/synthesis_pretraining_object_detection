import argparse
import os

from bs4 import BeautifulSoup
import cv2
import imgaug as ia
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import random
import string
import sys
import tensorflow as tf
import torch
from tqdm import tqdm

from augmentation import photometric_seq, geometric_seq
from birds_loader import load_birds

root_path = os.path.abspath(os.path.join('..'))
if root_path not in sys.path:
    sys.path.append(root_path)

# Set CPU as available physical device for tensorflow
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# Allow PyTorch to use the GPU as it will run ReDO, which is heavier than what TF does
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()

parser.add_argument('--num_samples', default=10000, type=int,
                    help='Number of samples to generate.')
parser.add_argument('--output_dir', default='../data/birds_fake_nogan')
parser.add_argument('--image_size', default=320, type=int,
                    help='Size of the (square) images.')
parser.add_argument('--min_scale', default=0.5, type=float,
                    help='Minimum proportion of the object boxes with respect \
                    to the image size.')
parser.add_argument('--max_scale', default=0.9, type=float,
                    help='Maximum proportion of the object boxes with respect \
                    to the image size.')
parser.add_argument('--min_aspect_ratio', default=0.5, type=float,
                    help='Minimum aspect ratio for the boxes (w / h).')
parser.add_argument('--max_aspect_ratio', default=2.0, type=float,
                    help='Maximum aspect ratio for the boxes (w / h).')
parser.add_argument('--min_objects', default=1, type=int,
                    help='Minimum number of objects to place \
                    on a single image.')
parser.add_argument('--max_objects', default=3, type=int,
                    help='Maximum number of objects to place \
                    on a single image.')
parser.add_argument('--max_tries', default=10, type=int,
                    help='Maximum number of tries after sampling a number of objects \
                    before giving up and sampling another number.')
parser.add_argument('--voc_dir', default='../data/VOCdevkit', type=str,
                    help='Root directory for the Pascal VOC dataset, from which the \
                    background scenes will be taken.')
parser.add_argument('--real_objects_dir', default='../data/CUB_200_2011/CUB_200_2011', type=str,
                    help='Root directory for the images of real objects.')

args = parser.parse_args()
NUM_SAMPLES = args.num_samples
OUTPUT_DIR = args.output_dir
IMAGE_SIZE = args.image_size
MIN_SCALE = args.min_scale
MAX_SCALE = args.max_scale
MIN_SIZE = int(MIN_SCALE * IMAGE_SIZE)
MAX_SIZE = int(MAX_SCALE * IMAGE_SIZE)
MIN_ASPECT_RATIO = args.min_aspect_ratio
MAX_ASPECT_RATIO = args.max_aspect_ratio
MIN_OBJECTS_PER_IMAGE = args.min_objects
MAX_OBJECTS_PER_IMAGE = args.max_objects
MAX_TRIES = args.max_tries
VOC_DIR = args.voc_dir
REAL_OBJECTS_DIR = args.real_objects_dir
IGNORE_VOC_CLASSES = []

######### Object data loaders ########################################

from ReDO import models

states = torch.load('cub_nets_state.pth')
opt = states['options']
MASK_SIZE = opt.sizex

if "netEncM" in states:
    netEncM = models._netEncM(sizex=opt.sizex, nIn=opt.nx, nMasks=opt.nMasks, nRes=opt.nResM, nf=opt.nfM, temperature=opt.temperature).to(device)
    netEncM.load_state_dict(states["netEncM"])
    netEncM.eval()

real_image_paths, bbox_crops = load_birds(root=REAL_OBJECTS_DIR, split='train')

def sample_object():
    obj_idx = np.random.choice(len(real_image_paths))
    image_path, bbox = real_image_paths[obj_idx], bbox_crops[obj_idx][0]
    image = Image.open(image_path.numpy())
    image = image.crop(bbox.numpy()[:-1])
    image = image.resize((MASK_SIZE, MASK_SIZE))
    image = np.array(image)
    if image.ndim != 3: return sample_object()
    return image

def calculate_mask(obj):
    obj_t = obj.transpose(2, 0, 1)
    inp = torch.from_numpy(obj_t.astype('float32') / 255.0)
    inp = inp.unsqueeze(0).to(device)

    mask = netEncM(inp)[:, :1].cpu().detach()
    mask = mask.squeeze()
    mask = mask.numpy()
    
    return mask


######### Background data loaders ####################################


def get_voc_classes(xml_path):
    if not os.path.exists(xml_path):
        raise Exception('Annotation file %s not found' % xml_path)

    with open(xml_path, 'r') as f:
        annotation = BeautifulSoup(f, 'lxml')

    classes = []

    for obj in annotation.find_all('object'):
        label = obj.find('name').text
        classes.append(label)

    return classes


voc_image_paths = []
for year in ['VOC2007', 'VOC2012']:
    splits = ['train']
    if year == 'VOC2007':
        splits += ['test']

    ids = []
    for split in splits:
        ids_file = os.path.join(VOC_DIR, year, 'ImageSets', 'Main', split+'.txt')

        with open(ids_file, 'r') as f:
            ids += [line.strip() for line in f.readlines()]

        for id in tqdm(ids, desc='Loading {}-{}'.format(year, split)):
            ann_path = os.path.join(VOC_DIR, year, 'Annotations', id+'.xml')
            img_classes = get_voc_classes(ann_path)

            if not any(c in img_classes for c in IGNORE_VOC_CLASSES):
                img_path = os.path.join(VOC_DIR, year, 'JPEGImages', id+'.jpg')
                voc_image_paths.append(img_path)

print(len(voc_image_paths), 'background images available')


def read_and_resize_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.uint8)
    return image


background_data = tf.data.Dataset.from_tensor_slices(voc_image_paths)
background_data = background_data.repeat()
background_data = background_data.shuffle(100, reshuffle_each_iteration=True)
background_data = background_data.map(read_and_resize_image)

background_data_iter = iter(background_data)


def sample_background():
    global background_data_iter
    return next(background_data_iter).numpy()


######### Here is where the detection samples are really created  #####################

def draw_object(backg):
    obj = sample_object()
    mask = calculate_mask(obj)
    
    obj = photometric_seq(image=obj)
    det_aug = geometric_seq.to_deterministic()
    obj = det_aug(image=obj)
    mask = det_aug(image=mask)

    mask = (mask > 0.1)*1.0
    row_mask = mask.any(axis=1)
    col_mask = mask.any(axis=0)

    Xs, = np.where(col_mask)
    Ys, = np.where(row_mask)

    try:
        x1, y1 = Xs.min(), Ys.min()
        x2, y2 = Xs.max(), Ys.max()
    except:
        x1, y1, x2, y2 = 0, 0, backg.shape[1], backg.shape[0]

    x1 *= backg.shape[1] / obj.shape[1]
    x2 *= backg.shape[1] / obj.shape[1]
    y1 *= backg.shape[0] / obj.shape[0]
    y2 *= backg.shape[0] / obj.shape[0]
    obj = cv2.resize(obj, (backg.shape[1], backg.shape[0]))
    mask = cv2.resize(mask, (backg.shape[1], backg.shape[0]))

    if mask.ndim < 3:
        mask = np.expand_dims(mask, -1)

    return (backg*(1 - mask) + obj*mask).astype('uint8'), x1, y1, x2, y2


def create_sample():
    while True:
        num_objects = np.random.randint(MIN_OBJECTS_PER_IMAGE,
                                        MAX_OBJECTS_PER_IMAGE + 1)
        ok = False

        for _ in range(MAX_TRIES):
            bnd_boxes = []

            for _ in range(num_objects):
                bird_size = np.random.randint(MIN_SIZE, MAX_SIZE)
                bird_ar = np.random.uniform(MIN_ASPECT_RATIO, MAX_ASPECT_RATIO)
                bird_width = min(IMAGE_SIZE, int(bird_size * np.sqrt(bird_ar)))
                bird_height = min(IMAGE_SIZE, int(bird_size / np.sqrt(bird_ar)))

                y1 = np.random.randint(0, IMAGE_SIZE - bird_height + 1)
                x1 = np.random.randint(0, IMAGE_SIZE - bird_width + 1)
                y2 = y1 + bird_height
                x2 = x1 + bird_width
                bnd_boxes.append(
                    ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label='bird')
                )

            ok = True
            for i, box1 in enumerate(bnd_boxes):
                for j, box2 in enumerate(bnd_boxes):
                    if i != j and box1.intersection(box2):
                        ok = False

            if ok: break
        if ok: break

    image = sample_background()

    for box in bnd_boxes:
        y1 = box.y1_int
        x1 = box.x1_int
        y2 = box.y2_int
        x2 = box.x2_int

        backg_patch = image[y1:y2, x1:x2]
        image[y1:y2, x1:x2], nx1, ny1, nx2, ny2 = draw_object(backg_patch)

        box.x1, box.x2 = nx1 + x1, nx2 + x1
        box.y1, box.y2 = ny1 + y1, ny2 + y1

    bnd_boxes = ia.BoundingBoxesOnImage(bnd_boxes, shape=image.shape)
    return image, bnd_boxes


ANNOTATIONS_FILE = os.path.join(OUTPUT_DIR, 'birds_fake.csv')
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

data = {
    'image_id': [],
    'xmin': [],
    'ymin': [],
    'xmax': [],
    'ymax': [],
    'label': []
}

for idx in tqdm(range(NUM_SAMPLES), desc='Generating dataset...'):
    image, bnd_boxes = create_sample()
    image_id = '%05d' % (idx + 1)

    image = Image.fromarray(image)
    image.save(os.path.join(IMAGES_DIR, image_id+'.jpg'))

    for box in bnd_boxes.bounding_boxes:
        data['image_id'].append(image_id)
        data['xmin'].append(box.x1_int)
        data['ymin'].append(box.y1_int)
        data['xmax'].append(box.x2_int)
        data['ymax'].append(box.y2_int)
        data['label'].append(box.label)

data = pd.DataFrame(data, columns=['image_id', 'xmin', 'ymin',
                                   'xmax', 'ymax', 'label'])
data.to_csv(ANNOTATIONS_FILE, index=False)
