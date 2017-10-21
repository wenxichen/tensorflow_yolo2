"""ILSVRC 2017 Classicifation Dataset.
"""

import os
from scipy import misc
import math
import numpy as np
import random
import pickle
import xml.etree.ElementTree as ET
from tqdm import trange, tqdm
from multiprocessing import Process, Array, Queue

import config as cfg


class ilsvrc_cls:

    def __init__(self, image_set, rebuild=False,
                 multithread=False, batch_size=cfg.BATCH_SIZE,
                 image_size = cfg.IMAGE_SIZE, random_noise=False):
        self.name = 'ilsvrc_2017_cls'
        self.devkit_path = cfg.ILSVRC_PATH
        self.data_path = self.devkit_path
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_set = image_set
        self.rebuild = rebuild
        self.multithread = multithread
        self.random_noise = random_noise
        self.load_classes()
        self.cursor = 0
        self.epoch = 1
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
        self.image_num = len(gt_labels)
        self.total_batch = int(math.ceil(self.image_num / float(self.batch_size)))


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
                imname)
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
        self.read_batch_array_size = self.total_batch + self.prefetch_size * self.batch_size
        self.readed_batch = Array('i', self.read_batch_array_size)
        for i in range(self.read_batch_array_size):
            self.readed_batch[i] = 0
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

        # reset after one epoch
        if self.batch_cursor_read == self.total_batch:
            self.num_batch_left = -1
            self.epoch += 1
            self.cursor = 0
            self.batch_cursor_read = 0
            self.batch_cursor_fetched = 0
            random.shuffle(self.gt_labels)
            for i in range(self.read_batch_array_size):
                self.readed_batch[i] = 0
            print "######### reset, epoch", self.epoch, "start!########"
            # prefill the fetch task for the new epoch
            for i in range(self.num_child):
                self.start_prefetch(i)
            for i in range(self.num_child / 2):
                self.collect_prefetch(i)

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
                    imname)
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

    def add_4_side_contrast(self, x):
        """This is deprecated, please use add_4_side_contrast_mtr."""
        x_with_contrast = np.zeros([299, 299, 3])
        for r in range(299):
            for c in range(299):
                for i in range(3):
                    x_with_contrast[r,c,i] = x[r,c,i]
                    if r != 0:
                        x_with_contrast[r,c, 3+i] = abs(x[r,c,i] - x[r-1,c,i])
                    if r != 299 - 1:
                        x_with_contrast[r,c, 3*2+i] = abs(x[r,c,i] - x[r+1,c,i])
                    if c != 0:
                        x_with_contrast[r,c, 3*3+i] = abs(x[r,c,i] - x[r,c-1,i])
                    if c != 299 - 1:
                        x_with_contrast[r,c, 3*4+i] = abs(x[r,c,i] - x[r,c+1,i])
        return x_with_contrast

    def add_4_side_contrast_mtr(self, x):
        """Add the contrast of the 4 sides of each pixel value for each rgb channel.
        Thus, for each rgb channel, 4 new channels is created s.t. for pixle (r,c):
            1) absolute difference between (r,c) and (r-1,c)
            2) absolute difference between (r,c) and (r+1,c)
            3) absolute difference between (r,c) and (r,c-1)
            4) absolute difference between (r,c) and (r,c+1)
        """
        x_with_contrast = np.zeros([299, 299, 3])
        x_with_contrast[:,:,0:3] = x
        x_with_contrast[1:,:,3:6] = np.abs(x[1:,:,:] - x[:-1,:,:])
        x_with_contrast[:-1,:,6:9] = np.abs(x[:-1,:,:] - x[1:,:,:])
        x_with_contrast[:,1:,9:12] = np.abs(x[:,1:,:] - x[:,:-1,:])
        x_with_contrast[:,:-1,12:] = np.abs(x[:,:-1,:] - x[:,1:,:])
        return x_with_contrast

    def image_read(self, imname):
        image = misc.imread(imname, mode='RGB').astype(np.float)
        r,c,ch = image.shape
        if r < 299 or c < 299:
            # TODO: check too small images
            # print "##too small!!"
            image = misc.imresize(image, (299, 299, 3))
        elif r > 299 or c > 299:
            image = image[(r-299)/2 : (r-299)/2 + 299, (c-299)/2 : (c-299)/2 + 299, :]
        # print r, c, image.shape
        assert image.shape == (299, 299, 3)
        image = (image / 255.0) * 2.0 - 1.0
        if self.random_noise:
            add_noise = bool(random.getrandbits(1))
            if add_noise:
                eps = random.choice([4.0, 8.0, 12.0, 16.0]) / 255.0 * 2.0
                noise_image = image + eps * np.random.choice([-1, 1], (299,299,3))
                image = np.clip(noise_image, -1.0, 1.0)
        return image


def save_synset_to_ilsvrcid_map(meta_file):
    """Create a map from synset to ilsvrcid and save it as a pickle file.
    """

    from scipy.io import loadmat
    meta = loadmat(meta_file)

    D = {}
    for item in meta['synsets']:
        D[str(item[0][1][0])] = item[0][0][0,0]
    
    pickle_file = os.path.join(os.path.dirname(__file__), 'syn2ilsid_map.pickle')
    with open(pickle_file, 'wb') as f:
        pickle.dump(D, f)


def save_ilsvrcid_to_synset_map(meta_file):
    """Create a map from ilsvrcid to synset and save it as a pickle file.
    """

    from scipy.io import loadmat
    meta = loadmat(meta_file)

    D = {}
    for item in meta['synsets']:
        D[item[0][0][0,0]] = str(item[0][1][0]) 
    
    pickle_file = os.path.join(os.path.dirname(__file__), 'ilsid2syn_map.pickle')
    with open(pickle_file, 'wb') as f:
        pickle.dump(D, f)

