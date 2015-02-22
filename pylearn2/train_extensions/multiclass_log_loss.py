"""
TrainExtension subclass for calculating multiclass log loss on monitoring
dataset(s), reported via monitor channels.
"""

__author__ = "Jotham Apaloo"
__copyright__ = "Copyright 2015"
__license__ = "TBD"
__maintainer__ = "Jotham Apaloo"

import numpy as np
import theano
import pdb
from theano import gof, config
from theano import tensor as T

from pylearn2.train_extensions import TrainExtension

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples, n_classes]
            true class, column index with unit value = true 
            class for sample
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    pdb.set_trace()
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = y_true
    n_samples = actual.shape[0]
    #actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(y_true * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss


class MulticlassLogLossOp(gof.Op):
    """
    Brief description.

    Parameters
    ----------
    name : str, optional (default 'mll')
        Name of this Op.
    use_c_code : WRITEME
    """
    def __init__(self, name='mll', use_c_code=theano.config.cxx):
        super(MulticlassLogLossOp, self).__init__(use_c_code)
        self.name = name

    def make_node(self, y_true, y_score):
        """
        Calculate mll.

        Parameters
        ----------
        y_true : tensor_like
            Target class labels.
        y_score : tensor_like
            Predicted probabilities for all classes class.
        """
        y_true = T.as_tensor_variable(y_true)
        y_score = T.as_tensor_variable(y_score)
        output = [T.scalar(name=self.name, dtype=config.floatX)]
        return gof.Apply(self, [y_true, y_score], output)

    def perform(self, node, inputs, output_storage):
        """
        Calculate mll.

        Parameters
        ----------
        node : Apply instance
            Symbolic inputs and outputs.
        inputs : list
            Sequence of inputs.
        output_storage : list
            List of mutable 1-element lists.
        """
        y_true, y_score = inputs
        try:
            mll = multiclass_log_loss(y_true, y_score)
        except ValueError:
            mll = np.nan
        output_storage[0][0] = theano._asarray(mll, dtype=config.floatX)


class MLLChannel(TrainExtension):
    """
    Adds a mll channel to the monitor for each monitoring dataset.

    This monitor MAY return nan unless all classes are represented in
    y_true. For this reason, it is recommended to set monitoring_batches
    to 1, especially when using unbalanced datasets.

    Parameters
    ----------
    channel_name_suffix : str, optional (default 'mll')
        Channel name suffix.
    positive_class_index : int, optional (default 1)
        Index of positive class in predicted values.
    negative_class_index : int or None, optional (default None)
        Index of negative class in predicted values for calculation of
        one vs. one performance. If None, uses all examples not in the
        positive class (one vs. the rest).
    """
    def __init__(self, channel_name_suffix='mll', positive_class_index=1,
                 negative_class_index=None):
        self.channel_name_suffix = channel_name_suffix
        self.positive_class_index = positive_class_index
        self.negative_class_index = negative_class_index

    def setup(self, model, dataset, algorithm):
        """
        Add mll channels for monitoring dataset(s) to model.monitor.

        Parameters
        ----------
        model : object
            The model being trained.
        dataset : object
            Training dataset.
        algorithm : object
            Training algorithm.
        """
        m_space, m_source = model.get_monitoring_data_specs()
        state, target = m_space.make_theano_batch()

        y = target
        y_hat = model.fprop(state)

        mll = MulticlassLogLossOp(self.channel_name_suffix)(y, y_hat)
        mll = T.cast(mll, config.floatX)

        for dataset_name, dataset in algorithm.monitoring_dataset.items():
            if dataset_name:
                channel_name = '{0}_{1}'.format(dataset_name,
                                                self.channel_name_suffix)
            else:
                channel_name = self.channel_name_suffix
            model.monitor.add_channel(name=channel_name,
                                      ipt=(state, target),
                                      val=mll,
                                      data_specs=(m_space, m_source),
                                      dataset=dataset)
