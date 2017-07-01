"""Train ILSVRC2017 Data using homemade scripts."""

import cv2

import config as cfg
from img_dataset.ilsvrc2017_cls import ilsvrc_cls

imdb = ilsvrc_cls('train')

