"""ILSVRC 2017 Classicifation Dataset.
"""

import os
import cv2
import math
import numpy as np
import random
import pickle
import xml.etree.ElementTree as ET
from tqdm import trange, tqdm
from multiprocessing import Process, Array, Queue

import config as cfg


class ilsvrc_cls:

    def __init__(self, image_set, rebuild=False, data_aug=False, multithread=False):
        self.name = 'ilsvrc_2017_cls'
        self.devkit_path = cfg.ILSVRC_PATH
        self.data_path = self.devkit_path
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.image_set = image_set
        self.rebuild = rebuild
        self.multithread = multithread
        self.data_aug = data_aug
        self.load_classes()
        self.cursor = 0
        self.epoch = 1
        # TODO: not hard code totoal number of images
        self.total_batch = int(math.ceil(1281167 / float(self.batch_size)))
        # self.total_batch = 70
        self.gt_labels = None
        assert os.path.exists(self.devkit_path), \
            'ILSVRC path does not exist: {}'.format(self.devkit_path)
        assert os.path.exists(self.data_path), \
            'Path does not exist: {}'.format(self.data_path)
        self.prepare()

        if self.multithread:
            self.prepare_multithread()
            self.get = self._get_multithread
        else:
            self.get = self._get

    def prepare(self):
        """Create a list of ground truth that includes input path and label.
        """
        # TODO: may still need to implement test
        cache_file = os.path.join(
            self.cache_path, 'ilsvrc_cls_' + self.image_set + '_gt_labels.pkl')
        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            print('{} {} dataset gt_labels loaded from {}'.
                  format(self.name, self.image_set, cache_file))
        else:
            if (self.image_set == "train"):
                imgset_fname = "train_cls.txt"
            else:
                imgset_fname = self.image_set + ".txt"
            imgset_file = os.path.join(
                self.data_path, 'ImageSets', 'CLS-LOC', imgset_fname)
            anno_dir = os.path.join(
                self.data_path, 'Annotations', 'CLS-LOC', self.image_set)
            print('Processing gt_labels using ' + imgset_file)
            gt_labels = []
            with open(imgset_file, 'r') as f:
                for line in tqdm(f.readlines()):
                    img_path = line.strip().split()[0]
                    if (self.image_set == "train"):
                        label = self.class_to_ind[img_path.split("/")[0]]
                    else:
                        anno_file = os.path.join(anno_dir, img_path + '.xml')
                        tree = ET.parse(anno_file)
                        label = tree.find('object').find('name').text
                        label = self.class_to_ind[label]
                    imname = os.path.join(
                        self.data_path, 'Data', 'CLS-LOC', self.image_set, img_path + ".JPEG")
                    gt_labels.append(
                        {'imname': imname, 'label': label})
            print('Saving gt_labels to: ' + cache_file)
            with open(cache_file, 'wb') as f:
                pickle.dump(gt_labels, f)
        random.shuffle(gt_labels)
        self.gt_labels = gt_labels

    def _get(self):
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
            images[count, :, :, :] = self.image_read(
                imname, data_aug=self.data_aug)
            labels[count] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def prepare_multithread(self):
        """Preperation for mutithread processing."""

        self.reset = False
        # num_batch_left should always be -1 until the last batch block of the epoch
        self.num_batch_left = -1
        self.num_child = 10
        self.child_processes = [None] * self.num_child
        self.batch_cursor_read = 0
        self.batch_cursor_fetched = 0
        # TODO: add this to cfg file
        self.prefetch_size = 5  # in terms of batch
        # TODO: may not need readed_batch after validating everything
        self.readed_batch = Array('i', self.total_batch)
        self.prefetched_images = np.zeros((self.batch_size * self.prefetch_size
                                           * self.num_child,
                                           self.image_size, self.image_size, 3))
        self.prefetched_labels = np.zeros(
            (self.batch_size * self.prefetch_size * self.num_child))
        self.queue_in = []
        self.queue_out = []
        for i in range(self.num_child):
            self.queue_in.append(Queue())
            self.queue_out.append(Queue())
            self.start_process(i)
            self.start_prefetch(i)

        # fetch the first one
        desc = 'receive the first half: ' + \
            str(self.num_child * self.prefetch_size / 2) + ' batches'
        for i in trange(self.num_child / 2, desc=desc):
            #     print "collecting", i
            self.collect_prefetch(i)

    def start_process(self, n):
        """Start multiprocessing prcess n."""
        self.child_processes[n] = Process(target=self.prefetch,
                                          args=(self.readed_batch,
                                                self.queue_in[n],
                                                self.queue_out[n]))
        self.child_processes[n].start()

    def start_prefetch(self, n):
        """Start prefetching in process n."""
        self.queue_in[n].put([self.cursor + self.batch_size * n * self.prefetch_size,
                              self.batch_cursor_fetched + self.prefetch_size * n])

        # maintain cusor and batch_cursor_fetched here
        # so it is easier to syncronize between threads
        if n == self.num_child - 1:
            batch_block = self.prefetch_size * self.num_child
            self.cursor += self.batch_size * batch_block
            self.batch_cursor_fetched += batch_block
            if self.total_batch <= self.batch_cursor_fetched + batch_block:
                self.reset = True
                self.num_batch_left = self.total_batch - self.batch_cursor_fetched
        # print "batch_cursor_fetched:", self.batch_cursor_fetched

    def start_prefetch_list(self, L):
        """Start multiple multiprocessing prefetches."""
        for p in L:
            self.start_prefetch(p)

    def collect_prefetch(self, n):
        """Collect prefetched data, join the processes.
        Join is not inculded because it seems faster to have
        Queue.get() perform in clusters.
        """
        images, labels = self.queue_out[n].get()
        fetch_size = self.batch_size * self.prefetch_size
        self.prefetched_images[n * fetch_size:(n + 1) * fetch_size] = images
        self.prefetched_labels[n * fetch_size:(n + 1) * fetch_size] = labels

    def collect_prefetch_list(self, L):
        """Collect and join a list of prefetcging processes."""
        for p in L:
            self.collect_prefetch(p)

    def close_all_processes(self):
        """Empty and close all queues, then terminate all child processes."""
        for i in range(self.num_child):
            self.queue_in[i].cancel_join_thread()
            self.queue_out[i].cancel_join_thread()
        for i in range(self.num_child):
            self.child_processes[i].terminate()

    def load_classes(self):
        """Use the folder name to get labels."""
        # TODO: double check if the classes are all the same as for train, test, val
        img_folder = os.path.join(
            self.data_path, 'Data', 'CLS-LOC', 'train')
        print('Loading class info from ' + img_folder)
        self.classes = [item for item in os.listdir(img_folder)
                        if os.path.isdir(os.path.join(img_folder, item))]
        self.num_class = len(self.classes)
        assert (self.num_class == 1000), "number of classes is not 1000!"
        self.class_to_ind = dict(
            list(zip(self.classes, list(range(self.num_class)))))

    def _get_multithread(self):
        """Get in multithread mode.
        Besides getting images and labels, 
        the function also manages start and end of child processes for prefetching data.

        Return: 
            images: 4D numpy array
            labels: 1D numpy array
        """

        # print "num_batch_left:", self.num_batch_left

        if self.reset:
            print "one epoch is about to finish! reseting..."
            self.collect_prefetch_list(
                range(self.num_child / 2, self.num_child))
            # print "#####passed here 1"
            self.reset = False

        elif self.num_batch_left == -1:
            # run the child process
            batch_block = self.prefetch_size * self.num_child
            checker = (self.batch_cursor_read % batch_block) - 4
            # print "checker:", checker
            if checker % 5 == 0:
                # print "about to start prefetch", checker / 5
                self.start_prefetch(int(checker / 5))
                if checker / 5 == self.num_child / 2 - 1:
                    self.collect_prefetch_list(
                        range(self.num_child / 2, self.num_child))

                elif checker / 5 == self.num_child - 1:
                    self.collect_prefetch_list(range(self.num_child / 2))

        assert (self.readed_batch[self.batch_cursor_read] == 1), \
            "batch not prefetched!"
        start_index = (self.batch_cursor_read
                       % (self.prefetch_size * self.num_child)) \
            * self.batch_size
        self.batch_cursor_read += 1
        # print "batch_cursor_read:", self.batch_cursor_read

        if self.num_batch_left == self.total_batch - self.batch_cursor_read:
            # fetch and receive the last few batches of the epoch
            L = range(int(math.ceil(self.num_batch_left /
                                    float(self.prefetch_size))))
            self.start_prefetch_list(L)
            self.collect_prefetch_list(L)
            # print "#####passed here 2"

        # reset after one epoch
        if self.batch_cursor_read == self.total_batch:
            self.num_batch_left = -1
            self.epoch += 1
            self.cursor = 0
            self.batch_cursor_read = 0
            self.batch_cursor_fetched = 0
            random.shuffle(self.gt_labels)
            # prefill the fetch task for the new epoch
            for i in range(self.num_child):
                self.start_prefetch(i)

        return (self.prefetched_images[start_index:start_index + self.batch_size],
                self.prefetched_labels[start_index:start_index + self.batch_size])

    def prefetch(self, readed_batch, q_in, q_out):
        """Prefetch data when task coming in from q_in 
        and sent out the images and labels from q_out.
        Uses in multithread processing.
        q_in send in [cursor, batch_cursor_fetched].
        """
        fetch_size = self.batch_size * self.prefetch_size

        while True:
            cursor, batch_cursor_fetched = q_in.get()
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

            q_out.put([images, labels])

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
