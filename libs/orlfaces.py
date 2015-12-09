from scipy import misc
import os

if __name__ == "__main__":
    import sys


    orlfaces = orlfaces_loader(sys.argv[1:])
    print "type(orlfaces.images)"
    print type(orlfaces.images) # expect <type 'numpy.ndarray'>
    print orlfaces.images.shape # expect tuple w\ (num imgs, img flat size)
    print "type(orlfaces.labels)"
    print type(orlfaces.labels) # expect <type 'numpy.ndarray'>
    print orlfaces.labels.shape # expect tuple w\ (num imgs, number of class aka number of path)
