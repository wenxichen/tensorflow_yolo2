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
        """Create a list of ground truth that includes input path and label.
        """

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
                    {'imname': imname, 'label': label})
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
            assert (self.num_class == 1000), "number of classes is not 1000!"
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
            images[count, :, :, :] = self.image_read(imname, data_aug=True)
            labels[count] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                random.shuffle(self.gt_labels)
                self.cursor = 0
        return images, labels

    def image_read(self, imname, data_aug=False):
        image = cv2.imread(imname)

        #####################
        # Data Augmentation #
        #####################
        if data_aug:
            flip = bool(random.getrandbits(1))
            rotate_deg = random.randint(0, 359)
            # 75% chance to do random crop
            # another 25% change in maintaining input at 224x224
            # this help simplify the input processing for test, val
            # TODO: can make multiscale test input later
            random_crop_chance = random.randint(0, 3)
            too_small = False
            color_pert = bool(random.getrandbits(1))

            if flip:
                image = image[:, ::-1, :]
            # assume color image
            rows, cols, _ = image.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate_deg, 1)
            image = cv2.warpAffine(image, M, (cols, rows))

            # color perturbation
            if color_pert:
                hue_shift_sign = bool(random.getrandbits(1))
                hue_shift = random.randint(0, 10)
                saturation_shift_sign = bool(random.getrandbits(1))
                saturation_shift = random.randint(0, 10)
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                # TODO: currently not sure what cv2 does to values 
                # that are larger than the maximum.
                # It seems it does not cut at the max 
                # nor normalize the whole by multiplying a factor.
                # need to expore this in more detail
                if hue_shift_sign:
                    hsv[:, :, 0] += hue_shift
                else:
                    hsv[:, :, 0] -= hue_shift
                if saturation_shift_sign:
                    hsv[:, :, 1] += saturation_shift
                else:
                    hsv[:, :, 1] -= saturation_shift
                image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # random crop
            if random_crop_chance > 0:
                # current random crop upbound is 292 (1.3 x 224)
                short_side_len = random.randint(
                    self.image_size, cfg.RAND_CROP_UPBOUND)
                short_side = min([cols, rows])
                if short_side == cols:
                    scaled_cols = short_side_len
                    factor = float(short_side_len) / cols
                    scaled_rows = int(rows * factor)
                else:
                    scaled_rows = short_side_len
                    factor = float(short_side_len) / rows
                    scaled_cols = int(cols * factor)
                # print "scaled_cols and rows:", scaled_cols, scaled_rows
                if scaled_cols < 224 or scaled_rows < 224:
                    too_small = True
                    print "Image is too small,", imname
                else:
                    image = cv2.resize(image, (scaled_cols, scaled_rows))
                    col_offset = random.randint(0, scaled_cols - self.image_size)
                    row_offset = random.randint(0, scaled_rows - self.image_size)
                    # print "col_offset and row_offset:", col_offset, row_offset
                    image = image[row_offset:self.image_size + row_offset,
                                col_offset:self.image_size + col_offset]
                # assuming still using image size 224x224
                # print "image shape is", image.shape

            if random_crop_chance == 0 or too_small:
                image = cv2.resize(image, (self.image_size, self.image_size))

        else:
            image = cv2.resize(image, (self.image_size, self.image_size))

        image = image.astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0

        return image
