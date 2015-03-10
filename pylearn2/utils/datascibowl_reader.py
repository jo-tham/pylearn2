"""
Low-level utilities for reading in raw plankton images
"""

__author__ = "Jotham Apaloo"
__copyright__ = "TBD"
__credits__ = ["???"]
__license__ = ["???"]
__email__ = "jothamapaloo@gmail.com"
__maintainer__ = "Jotham Apaloo"

import glob
import os
import random
import pickle as pkl
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from pylearn2.utils.image_proc import random_transform


def read_datascibowl_images(path_to_imgs, maxPixel=32,
                            save_classnames=False,
                            classnames_dir=None, augment_factor=None):
    """
    Read datascibowl images from original files and
    directory structure

    Parameters
    ----------
    path_to_imgs : str
        absolute path to directory containing directories
        corresponding to classes, with each directory
        containing observations (images) for each class

    maxPixel : int
        size of images, image will be rescaled to 
        maxPixel rows x maxPixel columns

    augment_factor : int
    

    Returns
    _______
    images : ndarray, shape (n_images, n_rows, n_cols)
        An image array with individual examples indexed along
        the first axis and the image dimensions along the 
        the second and third axis

    labels : 1 one-dimensional array continaing the labels
        strings
    """

    print "Looking for data in %s" %path_to_imgs
    assert os.path.isdir(path_to_imgs), "%s doesn't exist" %path_to_imgs

    directory_names = glob.glob(os.path.join(path_to_imgs, "*"))

    # Rescale the images and create the combined metrics
    # and training labels, get the total training images
    number_of_images = 0
    class_samples = {}

    for folder in directory_names:
        currentClass = os.path.split(folder)[-1]
        for fileNameDir in os.walk(folder):
            class_samples[currentClass] = 0
            for fileName in fileNameDir[2]:
                # Only read in the images
                if fileName[-4:] != ".jpg":
                    continue
                number_of_images += 1
                class_samples[currentClass] += 1

    class_n_max = max(class_samples.values())
    print "Found %s images" %number_of_images
    print "Highest class sample is %s observations" %class_n_max

    imageSize = maxPixel * maxPixel
    num_rows = number_of_images if augment_factor is None else \
               class_n_max * augment_factor * 121
    num_features = imageSize

    X = np.zeros((num_rows, num_features), dtype=float)
    y = np.zeros((num_rows), dtype=np.int)

    files = []
    i = 0    
    label = 0
    # List of string of class names
    namesClasses = list()

    if augment_factor is not None:
        random.seed(20150310)
    # Navigate through the list of directories
    for folder in directory_names:
        print "Reading files from %s" %folder

        currentClass = os.path.split(folder)[-1]
        class_n_images = class_samples[currentClass]
        # Still using this list for class names so order 
        # preserved when pickled
        namesClasses.append(currentClass)

        for fileNameDir in os.walk(folder):
            folder_images = \
                [img for img in fileNameDir[2] if img[-4:] == '.jpg']
            skipped_files = \
                [img for img in fileNameDir[2] if img[-4:] != '.jpg']
            im_root_dir = fileNameDir[0]
            #print 'Skipping files %s' %skipped_files

            j = 0
            for resample_i in xrange(class_n_max):
                nameFileImage = "{0}{1}{2}".format(im_root_dir,\
                                                   os.sep, \
                                                   folder_images[j])
                image = imread(nameFileImage, as_grey=True)
                files.append(fileName)
                image = resize(image, (maxPixel, maxPixel))

                if resample_i >= len(folder_images):
                    image = random_transform(image, 0.5)

                # Store the rescaled image pixels
                X[i, 0:imageSize] = np.reshape(image, (1, imageSize))

                if augment_factor is None and j==len(folder_images)-1:
                    break
                # go back to the first image of this class
                # it would be better to read it from memory instead
                # of disk
                j = 0 if j == len(folder_images)-1 else j+1

                # Store the classlabel
                y[i] = label
                i += 1
            label += 1

    if save_classnames:
        assert os.path.isdir(classnames_dir),\
            "%s doesn't exist" %classnames_dir
        classnames_path = os.path.join(classnames_dir,
                                       'pred_colnames_reader.pkl')
        with open(classnames_path, 'w') as f:
            pkl.dump(namesClasses, f)
                
    X = X.reshape(X.shape[0], maxPixel, maxPixel, 1)
    y = y.reshape(y.shape[0], 1)
    return X, y
