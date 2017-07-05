"""ILSVRC 2017 Classicifation Dataset.
"""

import os
import cv2
import math
import numpy as np
import random
from tqdm import trange
from multiprocessing import Process, Array, Queue

import config as cfg


class ilsvrc_cls:

    def __init__(self, image_set, rebuild=False, data_aug=True):
        self.name = 'ilsvrc_2017'
        self.devkit_path = cfg.ILSVRC_PATH
        self.data_path = self.devkit_path
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.image_set = image_set
        self.rebuild = rebuild
        self.data_aug = data_aug
        self.load_classes()
        # self.gt_labels = None
        assert os.path.exists(self.devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self.devkit_path)
        assert os.path.exists(self.data_path), \
            'Path does not exist: {}'.format(self.data_path)
        self.prepare()

        # multithreading
        self.reset = False
        # num_batch_left should always be -1 until the last batch block of the epoch
        self.num_batch_left = -1
        self.num_child = 20
        self.child_processes = [None] * self.num_child
        self.cursor = 0
        self.epoch = 1
        self.batch_cursor_read = 0
        self.batch_cursor_fetched = 0
        self.prefetch_size = 5  # in terms of batch
        # if fetch_toggle is T: fetch to the first half of the prefetched array
        # else: fetch to the second half of the prefetched array
        self.fetch_toggle = True
        # TODO: not hard code totoal number of images
        self.total_batch = int(math.ceil(1281167 / float(self.batch_size)))
        self.readed_batch = Array('i', self.total_batch)
        self.q = Queue()
        # twice the prefetch size to maintain continuous flow
        self.prefetched_images = np.zeros((self.batch_size * self.prefetch_size
                                           * self.num_child,
                                           self.image_size, self.image_size, 3))
        self.prefetched_labels = np.zeros(
            (self.batch_size * self.prefetch_size * self.num_child))
        # fetch the first one
        desc = 'first fetch ' + str(self.num_child * self.prefetch_size) +' batches'
        for i in trange(self.num_child, desc=desc):
            self.start_prefetch(i)
            self.collect_prefetch(i)
            self.child_processes[i].join()

    def start_prefetch(self, n):
        """Start multiprocessing prefetch."""

        batch_block = self.prefetch_size * self.num_child
        self.child_processes[n] = Process(target=self.prefetch,
                                          args=(self.readed_batch,
                                                self.q, 
                                                self.cursor 
                                                + self.batch_size * n * self.prefetch_size,
                                                self.batch_cursor_fetched
                                                + self.prefetch_size * n))
        self.child_processes[n].start()

        # maintain cusor and batch_cursor_fetched here 
        # so it is easier to syncronize between threads
        if n == self.num_child - 1:
            self.cursor += self.batch_size * batch_block
            self.batch_cursor_fetched += batch_block
            if self.total_batch <= self.batch_cursor_fetched + batch_block:
                self.reset = True
                self.num_batch_left = self.total_batch - self.batch_cursor_fetched

    def collect_prefetch(self, n):
        """Collect prefetched data, join the processes.
        Join is not inculded because it seems faster to have
        Queue.get() perform in clusters.
        """

        images, labels = self.q.get()
        # self.child_processes[n].join()
        fetch_size = self.batch_size * self.prefetch_size

        self.prefetched_images[n*fetch_size:(n+1)*fetch_size] = images
        self.prefetched_labels[n*fetch_size:(n+1)*fetch_size] = labels


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

        if self.reset:
            print "one epoch finished! reseting..."
            for c in range(self.num_child / 2, self.num_child):
                self.collect_prefetch(c)
                self.child_processes[c].join()
            for c in range(math.ceil(self.num_batch_left / float(self.prefetch_size))):
                self.collect_prefetch(c)
                self.child_processes[c].join()
            self.reset = False

        elif self.num_batch_left == -1:
            # run the child process
            block_size = self.prefetch_size * self.num_child
            checker = (self.batch_cursor_read % block_size) - 4 
            if checker % 5 == 0:
                self.start_prefetch(int(checker/5))
                if checker / 5 == self.num_child / 2 - 1:
                    for c in range(self.num_child / 2, self.num_child):
                        self.collect_prefetch(c)
                    for c in range(self.num_child / 2, self.num_child):
                        self.child_processes[c].join()
                elif checker / 5 == self.num_child - 1:
                    for c in range(self.num_child / 2):
                        self.collect_prefetch(c)
                    for c in range(self.num_child / 2):
                        self.child_processes[c].join()
            

        assert (self.readed_batch[self.batch_cursor_read]== 1), \
               "batch not prefetched!"
        start_index = (self.batch_cursor_read
                       % (self.prefetch_size * self.num_child)) \
                      * self.batch_size
        self.batch_cursor_read += 1
        if self.batch_cursor_read == self.num_batch_left:
            self.num_batch_left = -1
            self.epoch += 1
            self.cursor = 0
            self.batch_cursor_read = 0
            self.batch_cursor_fetched = 0
            random.shuffle(self.gt_labels)
        return (self.prefetched_images[start_index:start_index + self.batch_size],
                self.prefetched_labels[start_index:start_index + self.batch_size])

    def prefetch(self, readed_batch, q, cursor, batch_cursor_fetched):
        """Prefetch images.
        """
        # TODO: need to fix reset!
        fetch_size = self.batch_size * self.prefetch_size
        images = np.zeros(
            (fetch_size, self.image_size, self.image_size, 3))
        labels = np.zeros(fetch_size)
        count = 0
        while count < fetch_size:
            imname = self.gt_labels[cursor]['imname']
            images[count, :, :, :] = self.image_read(
                imname, data_aug=self.data_aug)
            labels[count] = self.gt_labels[cursor]['label']
            count += 1
            cursor += 1
            # to simplify the multithread reading
            # the last batch will padded with the images
            # from the beginning of the same list
            if cursor >= len(self.gt_labels):
                cursor = 0
        for i in range(batch_cursor_fetched, batch_cursor_fetched + self.prefetch_size):
            readed_batch[i] = 1

        q.put([images, labels])

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
