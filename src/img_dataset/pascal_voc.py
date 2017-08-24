"""Pascal VOC dataset class
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy

import config as cfg


class pascal_voc:
    def __init__(self, image_set, batch_size=cfg.BATCH_SIZE, rebuild=False):
        self.name = 'voc_2007'
        self.devkit_path = cfg.PASCAL_PATH
        self.data_path = os.path.join(self.devkit_path, 'VOC2007')
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = batch_size
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.S
        self.classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor')
        self.num_class = len(self.classes)
        self.class_to_ind = dict(
            list(zip(self.classes, list(range(self.num_class)))))
        self.flipped = cfg.FLIPPED
        self.image_set = image_set
        self.rebuild = rebuild
        self.cursor = 0
        self.gt_labels = None
        assert os.path.exists(self.devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self.devkit_path)
        assert os.path.exists(self.data_path), \
            'Path does not exist: {}'.format(self.data_path)
        self.prepare()

    def get(self):
        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros(
            (self.batch_size, self.cell_size, self.cell_size, 25))
        count = 0
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
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

    def prepare(self):
        gt_labels = self.load_labels()
        # TODO: consider adding flipped data into the saved cache's
        if self.flipped:
            print 'Appending horizontally-flipped training examples ...'
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] = gt_labels_cp[idx]['label'][:, ::-1, :]
                for i in xrange(self.cell_size):
                    for j in xrange(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = self.image_size - \
                                1 - gt_labels_cp[idx]['label'][i, j, 1]
            gt_labels += gt_labels_cp
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels

    def load_labels(self):
        cache_file = os.path.join(
            self.cache_path, 'pascal_' + self.image_set + '_gt_labels.pkl')

        if os.path.isfile(cache_file) and not self.rebuild:
            print 'Loading gt_labels from: ' + cache_file
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            print '{} gt_labels loaded from {}'.format(self.name, cache_file)
            return gt_labels

        print 'Processing gt_labels from: ' + self.data_path

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        txtname = os.path.join(self.data_path, 'ImageSets', 'Main',
                               self.image_set + '.txt')
        assert os.path.exists(txtname), \
            'Path does not exist: {}'.format(txtname)
        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]

        gt_labels = []
        for index in self.image_index:
            label, num = self.load_pascal_annotation(index)
            if num == 0:
                continue
            imname = os.path.join(
                self.data_path, 'JPEGImages', index + '.jpg')
            gt_labels.append(
                {'imname': imname, 'label': label, 'flipped': False})
        print 'Saving gt_labels to: ' + cache_file
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels

    def load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """

        imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        im = cv2.imread(imname)
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]
        # im = cv2.resize(im, [self.image_size, self.image_size])

        label = np.zeros((self.cell_size, self.cell_size, 25))
        filename = os.path.join(
            self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        for obj in objs:
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = max(min((float(bbox.find('xmin').text) - 1)
                         * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1)
                         * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1)
                         * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1)
                         * h_ratio, self.image_size - 1), 0)
            cls_ind = self.class_to_ind[obj.find(
                'name').text.lower().strip()]
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + cls_ind] = 1

        return label, len(objs)
