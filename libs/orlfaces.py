from scipy import misc
import os
import numpy as np

class Data(object):
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], (
            "images.shape: %s labels.shape: %s" % (images.shape,
                                                   labels.shape))
        self._num_examples = images.shape[0]
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples


def orlfaces_loader(paths):
    nclass = len(paths)
    mtx_stack = list()
    print "orlfaces_loader: found %d class" % nclass ######
    for path in paths:
        print "orlfaces_loader: in [%s]" % path #####
        for i, filename in enumerate(os.listdir(path)):
            img = misc.imread(path + filename)
            mtx_stack.append(img)
    num_imgs = len(mtx_stack)
    rows = mtx_stack[0].shape[0]
    cols = mtx_stack[0].shape[1]
    channel = 1
    return np.concatenate(mtx_stack, axis=1).reshape(num_imgs, rows, cols, channel)
                


if __name__ == "__main__":
    import sys


    orlfaces = orlfaces_loader(sys.argv[1:])
    print orlfaces.shape
    print orlfaces[0].shape
    # print "type(orlfaces.images)"
    # print type(orlfaces.images) # expect <type 'numpy.ndarray'>
    # print orlfaces.images.shape # expect tuple w\ (num imgs, img flat size)
    # print "type(orlfaces.labels)"
    # print type(orlfaces.labels) # expect <type 'numpy.ndarray'>
    # print orlfaces.labels.shape # expect tuple w\ (num imgs, number of class aka number of path)
