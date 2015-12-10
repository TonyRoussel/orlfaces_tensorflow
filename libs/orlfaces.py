from scipy import misc
import os
import numpy as np


class Datas(object):
    pass

class Data(object):
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], (
            "images.shape: %s labels.shape: %s" % (images.shape,
                                                   labels.shape))
        self._num_classes = labels[0].shape[0]
        self._num_examples = images.shape[0]
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        self._num_inputs = images.shape[1]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        return

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_inputs(self):
        return self._num_inputs

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def orlfaces_loader(paths):
    nclass = len(paths)
    mtx_stack = list()
    lbl_stack = list()
    class_i = 0
    print "orlfaces_loader: found %d class" % nclass
    for path in paths:
        print "orlfaces_loader: in [%s]" % path #####
        for i, filename in enumerate(os.listdir(path)):
            img = misc.imread(path + filename)
            label = np.ndarray(shape=(nclass,))
            label[nclass- 1 - class_i] = 1.
            mtx_stack.append(img)
            lbl_stack.append(label)
        class_i = class_i + 1
    rows = mtx_stack[0].shape[0]
    cols = mtx_stack[0].shape[1]
    channel = 1

    train, test = split_list(zip(mtx_stack, lbl_stack), 70)
    imgs_train_stack, labels_train_stack = zip(*train)
    imgs_test_stack, labels_test_stack = zip(*test)
    
    imgs_train = np.concatenate(imgs_train_stack, axis=1).reshape(len(imgs_train_stack), rows, cols, channel)
    imgs_test = np.concatenate(imgs_test_stack, axis=1).reshape(len(imgs_test_stack), rows, cols, channel)
    labels_train = np.vstack(labels_train_stack)
    labels_test = np.vstack(labels_test_stack)
    datas = Datas()
    datas.train = Data(imgs_train, labels_train)
    datas.test = Data(imgs_test, labels_test)
    return datas


def split_list(mylist, percent):
    lft_sze = int(percent / 100. * len(mylist))
    rht_sze = len(mylist) - lft_sze
    lft = mylist[:lft_sze]
    rht = mylist[-rht_sze:]
    return (lft, rht)
                


if __name__ == "__main__":
    import sys


    orlfaces = orlfaces_loader(sys.argv[1:])
    print "orlfaces_loader returned", orlfaces.train.num_classes, orlfaces.train.num_inputs
    print "type(orlfaces.train.images)"
    print type(orlfaces.train.images) # expect <type 'numpy.ndarray'>
    print orlfaces.train.images.shape # expect tuple w\ (num imgs, img flat size)
    print "type(orlfaces.train.labels)"
    print type(orlfaces.train.labels) # expect <type 'numpy.ndarray'>
    print orlfaces.train.labels.shape # expect tuple w\ (num imgs, number of class aka number of path)
    print "type(orlfaces.test.images)"
    print type(orlfaces.test.images) # expect <type 'numpy.ndarray'>
    print orlfaces.test.images.shape # expect tuple w\ (num imgs, img flat size)
    print "type(orlfaces.test.labels)"
    print type(orlfaces.test.labels) # expect <type 'numpy.ndarray'>
    print orlfaces.test.labels.shape # expect tuple w\ (num imgs, number of class aka number of path)
