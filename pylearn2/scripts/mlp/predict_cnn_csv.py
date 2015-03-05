#!/usr/bin/env python
# coding: utf-8
"""
Script to predict values using a pkl model file.

This is a configurable script to make predictions.

Basic usage:

.. code-block:: none

    predict_csv.py pkl_file.pkl test.csv output.csv

Optionally it is possible to specify if the prediction is regression or
classification (default is classification). The predicted variables are
integer by default.
Based on this script: http://fastml.com/how-to-get-predictions-from-pylearn2/.
This script doesn't use batches. If you run out of memory it could be 
resolved by implementing a batch version.

"""
from __future__ import print_function

__authors__ = ["Jotham Apaloo"]
__license__ = "GPL"

import sys
import os
import argparse
import numpy as np
import pandas as pd
import pickle
import pdb

from pylearn2.utils import serial
from theano import tensor as T
from theano import function


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Launch a prediction from a pkl file"
    )
    parser.add_argument('model_filename',
                        help='Specifies the pkl model file')
    parser.add_argument('test_filename',
                        help='Specifies the csv file with the values to predict')
    parser.add_argument('output_filename',
                        help='Specifies the predictions output file')
    parser.add_argument('rownames_filename',
                        help='Specifies the row indices to use for the output file')
    parser.add_argument('colnames_filename',
                        help='Specifies the column names to use for the output file')
    parser.add_argument('--prediction_type', '-P',
                        default="regression",
                        help='Prediction type (classification/regression)')
    parser.add_argument('--output_type', '-T',
                        default="float",
                        help='Output variable type (int/float)')
    parser.add_argument('--has-headers', '-H',
                        dest='has_headers',
                        action='store_true',
                        help='Indicates the first row in the input file is feature labels')
    parser.add_argument('--has-row-label', '-L',
                        dest='has_row_label',
                        action='store_true',
                        help='Indicates the first column in the input file is row labels')
    parser.add_argument('--delimiter', '-D',
                        default=',',
                        help="Specifies the CSV delimiter for the test file. Usual values are \
                             comma (default) ',' semicolon ';' colon ':' tabulation '\\t' and space ' '")
    return parser

def predict(model_path, test_path, output_path, rownames_path, colnames_path,
            predictionType="classification", outputType="float",
            headers=False, first_col_label=False, delimiter=","):
    """
    Predict from a pkl file.

    Parameters
    ----------
    modelFilename : str
        The file name of the model file.
    testFilename : str
        The file name of the file to test/predict.
    outputFilename : str
        The file name of the output file.
    predictionType : str, optional
        Type of prediction (classification/regression).
    outputType : str, optional
        Type of predicted variable (int/float).
    headers : bool, optional
        Indicates whether the first row in the input file is feature labels
    first_col_label : bool, optional
        Indicates whether the first column in the input file is row labels (e.g. row numbers)
    """

    print("loading model...")

    try:
        model = serial.load(model_path)
    except Exception as e:
        print("error loading {}:".format(model_path))
        print(e)
        return False

    skiprows = 1 if headers else 0
    x = np.load(test_path)

    if first_col_label:
        x = x[:,1:]
    
    print("loading data and predicting...")
    # pdb.set_trace()
    
    print("setting up symbolic expressions...")
    # for layer in model.layers:
        # print("{0}:{1}".format(layer.layer_name, layer.get_param_values()))
    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)

    if predictionType == "classification":
        Y = T.argmax(Y, axis=1)

    f = function([X], Y, allow_input_downcast=True)

    y_pred = np.zeros((x.shape[0], 121), dtype=np.float32)
        
    # print "Shape of y_pred {}.".format(y_pred.shape)

    for i in xrange(x.shape[0]):
        y_pred[i, :] = f(x[[i], :, :, :])
    

    print("writing predictions...")

    with open(colnames_path, 'r') as colnames:
        labels = pickle.load(colnames)
    with open(rownames_path, 'r') as rownames:
        indices = pickle.load(rownames)
    df = pd.DataFrame(y_pred, columns=labels, index=indices)
    df.index.name = 'image'
    df.to_csv(output_path)
    return True

if __name__ == "__main__":
    """
    See module-level docstring for a description of the script.
    """
    parser = make_argument_parser()
    args = parser.parse_args()
    ret = predict(args.model_filename, args.test_filename, args.output_filename,
                  args.rownames_filename, args.colnames_filename,
                  args.prediction_type, args.output_type,
                  args.has_headers, args.has_row_label, args.delimiter)
    if not ret:
        sys.exit(-1)

