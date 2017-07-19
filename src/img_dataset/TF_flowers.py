"""ILSVRC 2017 Classicifation Dataset.
Use val_split to set the portion for validation set.
"""

import os
import cv2
import math
import numpy as np
import random
import pickle
import copy
from tqdm import trange, tqdm

import config as cfg


class tf_flowers:

    def __init__(self, val_split, rebuild=False, data_aug=False):
        self.name = 'TF_flowers'
        self.devkit_path = cfg.FLOWERS_PATH
        self.data_path = self.devkit_path
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.rebuild = rebuild
        self.data_aug = data_aug
        self.num_class = 5
        self.classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
        self.class_to_ind = dict(
            list(zip(self.classes, list(range(self.num_class)))))
        self.train_cursor = 0
        self.val_cursor = 0
        self.epoch = 1
        self.gt_labels = None
        self.val_split = val_split
        assert os.path.exists(self.devkit_path), \
            'TF_flowers path does not exist: {}'.format(self.devkit_path)
        assert os.path.exists(self.data_path), \
            'Path does not exist: {}'.format(self.data_path)
        self.prepare()

    def prepare(self):
        """Create a list of ground truth that includes input path and label.
        Then, split the data into training set and validation set according to val_split.
        """
        # TODO: may still need to implement test
        cache_file = os.path.join(
            self.cache_path, 'TF_flowers_gt_labels.pkl')
        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            print('{} dataset gt_labels loaded from {}'.
                  format(self.name, cache_file))
        else:
            print('Processing gt_labels using...')
            gt_labels = []
            for c in tqdm(self.classes):
                label = self.class_to_ind[c]
                c_data_dir = os.path.join(self.data_path, c)
                for f in os.listdir(c_data_dir):
                    if f[-4:].lower() == '.jpg':
                        imname = os.path.join(c_data_dir, f)
                        gt_labels.append({'imname': imname, 'label': label})
            print('Saving gt_labels to: ' + cache_file)
            with open(cache_file, 'wb') as f:
                pickle.dump(gt_labels, f)
        random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        self.dataset_size = len(gt_labels)
        self.total_batch = int(
            math.ceil(self.dataset_size / float(self.batch_size)))
        cut_idx = int(self.dataset_size * self.val_split)
        self.val_gt_labels = copy.deepcopy(gt_labels[:cut_idx])
        self.train_gt_labels = copy.deepcopy(gt_labels[cut_idx:])
        print('training set size: {:d}, validation set size: {:d}'
              .format(len(self.train_gt_labels), len(self.val_gt_labels)))

    def get_train(self):
        return self._get('train')

    def get_val(self):
        return self._get('val')

    def _get(self, image_set):
        """Get shuffled images and labels according to batchsize.
        Use image_set to set whether to get training set or validation set.
        validation set data will not have data augmentation.

        Return: 
            images: 4D numpy array
            labels: 1D numpy array
        """
        if image_set == 'val':
            gt_labels = self.val_gt_labels
            cursor = self.val_cursor
            data_aug = False
        elif image_set == 'train':
            gt_labels = self.train_gt_labels
            cursor = self.train_cursor
            data_aug = self.data_aug

        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros(self.batch_size)
        count = 0
        while count < self.batch_size:
            imname = gt_labels[cursor]['imname']
            images[count, :, :, :] = self.image_read(
                imname, data_aug=data_aug)
            labels[count] = gt_labels[cursor]['label']
            count += 1
            cursor += 1
            if cursor >= len(gt_labels):
                random.shuffle(self.train_gt_labels)
                cursor = 0
                if image_set == 'train':
                    self.epoch += 1

        if image_set == 'val':
            self.val_cursor = cursor
        elif image_set == 'train':
            self.train_cursor = cursor

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
            exposure_shift = bool(random.getrandbits(1))

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

            if exposure_shift:
                brighter = bool(random.getrandbits(1))
                if brighter:
                    gamma = random.uniform(1, 2)
                else:
                    gamma = random.uniform(0.5, 1)
                image = ((image / 255.0) ** (1.0 / gamma)) * 255

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
                    col_offset = random.randint(
                        0, scaled_cols - self.image_size)
                    row_offset = random.randint(
                        0, scaled_rows - self.image_size)
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
