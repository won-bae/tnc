import os
import glob
import json
import yaml
import logging
import numpy as np
from PIL import Image
from colorlog import ColoredFormatter

image_extensions = ['.jpg', '.jpeg', '.gif', '.png']

# Logging
# =======

def _infov(self, msg, *args, **kwargs):
    self.log(logging.INFO + 1, msg, *args, **kwargs)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'white,bold',
        'INFOV':    'cyan,bold',
        'WARNING':  'yellow',
        'ERROR':    'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
ch.setFormatter(formatter)

log = logging.getLogger('better')
log.setLevel(logging.DEBUG)
log.handlers = []       # No duplicated handlers
log.propagate = False   # workaround for duplicated logs in ipython
log.addHandler(ch)

logging.addLevelName(logging.INFO + 1, 'INFOV')
logging.Logger.infov = _infov


def setup(model_name, tag):
    return

def generate_tag(tag):
    if not tag:
        import random, string
        letters = string.ascii_lowercase
        tag = ''.join(random.choice(letters) for i in range(5))
        log.warn("Tag is not specified. Random tag '{}' is assigned".format(tag))
    else:
        log.warn("Tag '{}' is specified".format(tag))
    return tag

def load_config(config_name):
    root = os.getcwd()
    config_path = os.path.join(root, 'configs', config_name + '.yml')
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def load_json(image_dir, json_file):
    json_path = os.path.join(image_dir, json_file)
    with open(json_path, 'r') as f:
        metadata = json.load(f)

    image_paths = []
    labels = []

    for annotation in metadata['annotations']:
        image_path = os.path.join(image_dir, annotation['id'])
        ext = os.path.splitext(image_path)[1]
        if ext.lower() in image_extensions:
            label = int(annotation['category_id'])
            image_paths.append(image_path)
            labels.append(label)

    return (image_paths, labels)

def load_image_paths(image_dir, recursive=False):

    if recursive:
        file_paths = glob.glob(os.path.join(image_dir, '**', '*.*'), recursive=True)
    else:
        file_paths = glob.glob(os.path.join(image_dir, '*.*'))

    image_paths = []
    pseudo_labels = []

    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1]
        if ext.lower() in image_extensions:
            image_paths.append(file_path)
            pseudo_labels.append(-1)

    return (image_paths, pseudo_labels)

def load_images(image_paths):
    base_width, aspect_ratio = 1920, 0.75
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((base_width, int(base_width * aspect_ratio)))
        image = np.expand_dims(image, axis=0)
        images.append(image)
    return np.concatenate(images, axis=0)

def store_detections(images, boxes, scores, classes, threshold):
    return

def is_label_available(labels):
    return np.sum(labels == -1) <= 0
