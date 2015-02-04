"""
The 2015 datasciencebowl dataset.
"""
__authors__ = "Jotham Apaloo"
__copyright__ = "TBD"
__credits__ = ["Jotham Apaloo"]
__license__ = "TBD"
__maintainer__ = "Jotham Apaloo"
__email__ = "jothamapaloo@gmail.com"

import numpy as N
np = N
from theano.compat.six.moves import xrange
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import control
from pylearn2.datasets import cache
from pylearn2.utils import serial
from pylearn2.utils.datascibowl import read_datascibowl_images
from pylearn2.utils.rng import make_np_rng


class DataSciBowl(dense_design_matrix.DenseDesignMatrix):
    """
    The 2015 kaggle datascience bowl plankon dataset

    Parameters
    ----------

    """

    def __init__(self, axes=['b', 0, 1, 'c']):
        self.args = locals()

        def dimshuffle(b01c):
            """
            .. todo::

                WRITEME
            """
            default = ('b', 0, 1, 'c')
            return b01c.transpose(*[default.index(axis) for axis in axes])

        if control.get_load_data():
            path = "${PYLEARN2_DATA_PATH}/datascibowl/"
            im_path = path + 'train'
            topo_view, y = read_datascibowl_images(imgs_path)
            
        # if binarize:
        #     topo_view = (topo_view > 0.5).astype('float32')

        y_labels = 10

        # m, r, c = topo_view.shape
        # assert r == 28
        # assert c == 28
        topo_view = topo_view.reshape(m, r, c, 1)

        if which_set == 'train':
            assert m == 60000
        elif which_set == 'test':
            assert m == 10000
        else:
            assert False

        if center:
            topo_view -= topo_view.mean(axis=0)



        # here we just the vars to subclass denseDesignMatrix
        # see the DenseDesignMatrix class for their definition
        # topo_v
        super(MNIST, self).__init__(topo_view=dimshuffle(topo_view), y=y,
                                    axes=axes, y_labels=y_labels)

        assert not N.any(N.isnan(self.X))

        if start is not None:
            assert start >= 0
            if stop > self.X.shape[0]:
                raise ValueError('stop=' + str(stop) + '>' +
                                 'm=' + str(self.X.shape[0]))
            assert stop > start
            self.X = self.X[start:stop, :]
            if self.X.shape[0] != stop - start:
                raise ValueError("X.shape[0]: %d. start: %d stop: %d"
                                 % (self.X.shape[0], start, stop))
            if len(self.y.shape) > 1:
                self.y = self.y[start:stop, :]
            else:
                self.y = self.y[start:stop]
            assert self.y.shape[0] == stop - start

        if which_set == 'test':
            assert fit_test_preprocessor is None or \
                (fit_preprocessor == fit_test_preprocessor)

        if self.X is not None and preprocessor:
            preprocessor.apply(self, fit_preprocessor)
