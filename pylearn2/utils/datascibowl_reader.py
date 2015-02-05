"""
Low-level utilities for reading in raw plankton images
"""

__author__ = "Jotham Apaloo"
__copyright__ = "TBD"
__credits__ = ["???"]
__license__ = ["???"]
__email__ = "jothamapaloo@gmail.com"
__maintainer__ = "Jotham Apaloo"

import numpy as np
import glob
import os
from skimage.io import imread
from skimage.transform import resize


def read_datascibowl_images(path_to_imgs, maxPixel=32):
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
    numberofImages = 0
    for folder in directory_names:
        for fileNameDir in os.walk(folder):   
            for fileName in fileNameDir[2]:
                # Only read in the images
                if fileName[-4:] != ".jpg":
                    continue
                numberofImages += 1
    # print "Found %s images" %numberofImages

    imageSize = maxPixel * maxPixel
    num_rows = numberofImages
    num_features = imageSize

    X = np.zeros((num_rows, num_features), dtype=float)
    y = np.zeros((num_rows), dtype=np.int)

    files = []
    i = 0    
    label = 0
    # List of string of class names
    namesClasses = list()

    # Navigate through the list of directories
    for folder in directory_names:
        # Append the string class name for each class
        currentClass = folder.split(os.pathsep)[-1]
        namesClasses.append(currentClass)
        for fileNameDir in os.walk(folder):   
            for fileName in fileNameDir[2]:
                # Only read in the images
                if fileName[-4:] != ".jpg":
                    continue
            
            # Read in the images and create the features
            nameFileImage = "{0}{1}{2}".format(fileNameDir[0],\
                                               os.sep, fileName)            
            image = imread(nameFileImage, as_grey=True)
            files.append(nameFileImage)
            image = resize(image, (maxPixel, maxPixel))
            
            # Store the rescaled image pixels and the axis ratio
            X[i, 0:imageSize] = np.reshape(image, (1, imageSize))
            
            # Store the classlabel
            y[i] = label
            i += 1
            label += 1

    X = X.reshape(X.shape[0], 32, 32, 1)
    y = y.reshape(y.shape[0], 1)

    return X, y
