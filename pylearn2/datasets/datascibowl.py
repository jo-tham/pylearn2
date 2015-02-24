"""
The 2015 datasciencebowl dataset.
"""
__authors__ = "Jotham Apaloo"
__copyright__ = "TBD"
__credits__ = ["Jotham Apaloo"]
__license__ = "TBD"
__maintainer__ = "Jotham Apaloo"
__email__ = "jothamapaloo@gmail.com"

import os
import numpy as np
from pylearn2.utils.datascibowl_reader import read_datascibowl_images
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.rng import make_np_rng

class DataSciBowl(dense_design_matrix.DenseDesignMatrix):
    """
    The 2015 kaggle datascience bowl plankon dataset

    Parameters
    ----------
    which_set : str
        'train' or 'test'
        'test' currently doesnt work, these are for prediction
        only, response class is not known
    start :
    stop : 
    axes : 
    binarize : pixels with value < 0.5 assigned 1, else 0
    centre : centre pixel values

    """

    def __init__(self, axes=('b', 0, 1, 'c'), center=False,
                 binarize=False, start=None, stop=None,
                 which_set='train', shuffle=False):

        if which_set not in ['train']:
            if which_set == 'test':
                raise ValueError(
                    "test is not implemented, those data are for "
                    "prediction only.")
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train"].')

        
        path = os.path.expandvars("${PYLEARN2_DATA_PATH}/datascibowl/")
        imgs_path = path + which_set
        
        topo_view, y = read_datascibowl_images(imgs_path, 32)
            
        if binarize:
            topo_view = (topo_view > 0.5).astype('float64')

        if center:
            topo_view -= topo_view.mean(axis=0)

        y_labels = 121
        m = topo_view.shape[0]

        print 'topo_view.shape: {}'.format(topo_view.shape)
        print 'NAs: {}'.format(np.count_nonzero(np.isnan(topo_view)))

        if shuffle:
            self.shuffle_rng = make_np_rng(
                None, [1, 2, 3], which_method="shuffle")
            for i in xrange(topo_view.shape[0]):
                j = self.shuffle_rng.randint(m)
                if i % 1000 == 0:
                    print "Swapping %s and %s" % (i, j)

                # Copy ensures that memory is not aliased.
                tmp = topo_view[i, :, :, :].copy()
                topo_view[i, :, :, :] = topo_view[j, :, :, :]
                topo_view[j, :, :, :] = tmp

                tmp = y[i:i + 1].copy()
                y[i] = y[j]
                y[j] = tmp

        if start == None:
            start = 0
        if stop == None:
            stop = topo_view.shape[0]

        topo_view = topo_view[start:stop, :]
        y = y[start:stop, :]
        
        super(DataSciBowl, self).__init__(topo_view=topo_view,
                                          y=y,
                                          axes=axes,
                                          y_labels=y_labels)
