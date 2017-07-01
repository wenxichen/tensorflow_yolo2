"""ILSVRC 2017 Classicifation Dataset.
"""

import os
import cv2
import numpy as np
import random

import config as cfg


class ilsvrc_cls:

    def __init__(self, image_set, rebuild=False):
        self.name = 'ilsvrc_2017'
        self.devkit_path = cfg.ILSVRC_PATH
        self.data_path = self.devkit_path
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.flipped = cfg.FLIPPED
        self.image_set = image_set
        self.rebuild = rebuild
        self.cursor = 0
        self.load_classes()
        # self.gt_labels = None
        assert os.path.exists(self.devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self.devkit_path)
        assert os.path.exists(self.data_path), \
            'Path does not exist: {}'.format(self.data_path)
        self.prepare()

    def prepare(self):
        """Create a list of ground truth that includes input path, lable 
        and whether the image is flipped.
        """
        # TODO: consider adding flipped data and also put them into the saved cache's

        if (self.image_set == "train"):
            imgset_fname = "train_cls.txt"
        else:
            imgset_fname = self.image_set + ".txt"
        imgset_file = os.path.join(
            self.data_path, 'ImageSets', 'CLS-LOC', imgset_fname)
        print('Processing gt_labels using ' + imgset_file)
        gt_labels = []
        with open(imgset_file, 'r') as f:
            for line in f.readlines():
                img_path = line.strip().split()[0]
                label = self.class_to_ind[img_path.split("/")[0]]
                imname = os.path.join(
                    self.data_path, 'Data', 'CLS-LOC', self.image_set, img_path + ".JPEG")
                gt_labels.append(
                    {'imname': imname, 'label': label, 'flipped': False})
        random.shuffle(gt_labels)
        self.gt_labels = gt_labels

    def load_classes(self):
        """Use the folder name to get labels."""
        if (self.image_set == "train"):
            img_folder = os.path.join(
                self.data_path, 'Data', 'CLS-LOC', 'train')
            print('Loading class info from ' + img_folder)
            self.classes = [item for item in os.listdir(img_folder)
                            if os.path.isdir(os.path.join(img_folder, item))]
            self.num_class = len(self.classes)
            assert (self.num_class == 1000)
            self.class_to_ind = dict(
                list(zip(self.classes, list(range(self.num_class)))))

    def get(self):
        """Get shuffled images and labels according to batchsize.

        Return: 
            images: 4D numpy array
            labels: 1D numpy array
        """
        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros(self.batch_size)
        count = 0
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            # TODO: implement flip
            flipped = self.gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped)
            labels[count] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                random.shuffle(self.gt_labels)
                self.cursor = 0
        return images, labels

    def image_read(self, imname, flipped=False):
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
        return image
