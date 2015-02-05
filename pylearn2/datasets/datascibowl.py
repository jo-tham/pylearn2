"""
The 2015 datasciencebowl dataset.
"""
__authors__ = "Jotham Apaloo"
__copyright__ = "TBD"
__credits__ = ["Jotham Apaloo"]
__license__ = "TBD"
__maintainer__ = "Jotham Apaloo"
__email__ = "jothamapaloo@gmail.com"

import numpy as np
from pylearn2.utils.datascibowl_reader import read_datascibowl_images
from pylearn2.datasets import dense_design_matrix

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
                 which_set='train'):

        if which_set not in ['train']:
            if which_set == 'test':
                raise ValueError(
                    "test is not implemented, those data are for "
                    "prediction only.")
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train"].')

        
        path = "/home/joth/projects/2014-12-20_datascibowl/competition_data/"
        imgs_path = path + which_set
        
        topo_view, y = read_datascibowl_images(imgs_path, 32)
            
        if binarize:
            topo_view = (topo_view > 0.5).astype('float64')
        if center:
            topo_view -= topo_view.mean(axis=0)

        if start == None:
            start = 0
        if stop == None:
            stop = topo_view.shape[0]

        topo_view = topo_view[start:stop, :]
        y = y[start:stop, :]

        y_labels = 121

        print 'topo_view.shape: {}'.format(topo_view.shape)
        print 'NAs: {}'.format(np.count_nonzero(np.isnan(topo_view)))
        
        super(DataSciBowl, self).__init__(topo_view=topo_view,
                                          y=np.atleast_2d(y).T,
                                          axes=axes,
                                          y_labels=y_labels)
