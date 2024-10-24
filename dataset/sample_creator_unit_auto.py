'''
DL models (FNN, 1D CNN and CNN-LSTM) evaluation on N-CMAPSS
12.07.2021
Hyunho Mo
hyunho.mo@unitn.it
'''
## Import libraries in python
import gc
import argparse
import os
import json
import logging
import sys
import h5py
import time
import matplotlib
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
import random
import importlib
from scipy.stats import randint, expon, uniform
import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
# from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy.stats as stats


from utils.data_preparation_unit import df_all_creator, df_train_creator, df_test_creator, Input_Gen


seed = 0
random.seed(0)
np.random.seed(seed)


DS01_filename = 'N-CMAPSS_DS01-005.h5'
DS02_filename = 'N-CMAPSS_DS02-006.h5'
DS03_filename = 'N-CMAPSS_DS03-012.h5'
DS04_filename = 'N-CMAPSS_DS04.h5'
DS05_filename = 'N-CMAPSS_DS05.h5'
DS06_filename = 'N-CMAPSS_DS06.h5'
DS07_filename = 'N-CMAPSS_DS07.h5'
DS08_filename = 'N-CMAPSS_DS08a-009.h5' 
DS08c_filename = 'N-CMAPSS_DS08c-008.h5'

current_dir = os.path.dirname(os.path.abspath(__file__))
data_filedir = os.path.join(current_dir, 'dataset/DS08c')
data_filepath = os.path.join(current_dir, 'dataset', DS08c_filename)




def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='sample creator')
    parser.add_argument('-w', type=int, default=10, help='window length', required=True)
    parser.add_argument('-s', type=int, default=10, help='stride of window')
    parser.add_argument('--sampling', type=int, default=1, help='sub sampling of the given data. If it is 10, then this indicates that we assumes 0.1Hz of data collection')
    parser.add_argument('--test', type=int, default='non', help='select train or test, if it is zero, then extract samples from the engines used for training')

    args = parser.parse_args()

    sequence_length = args.w
    stride = args.s
    sampling = args.sampling
    selector = args.test



    # Load data
    '''
    W: operative conditions (Scenario descriptors)
    X_s: measured signals
    X_v: virtual sensors
    T(theta): engine health parameters
    Y: RUL [in cycles]
    A: auxiliary data
    '''

    df_all = df_all_creator(data_filepath, sampling)

    '''
    Split dataframe into Train and Test
    Training units: 2, 5, 10, 16, 18, 20
    Test units: 11, 14, 15

    '''
    # units = list(np.unique(df_A['unit']))
    units_index_train = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    # units_index_train = [2., 5., 10., 11., 14., 15., 16., 18., 20.,]
    units_index_test = [9.0, 10.0]

    print("units_index_train", units_index_train)
    print("units_index_test", units_index_test)

    # if any(int(idx) == unit_index for idx in units_index_train):
    #     df_train = df_train_creator(df_all, units_index_train)
    #     print(df_train)
    #     print(df_train.columns)
    #     print("num of inputs: ", len(df_train.columns) )
    #     df_test = pd.DataFrame()
    #
    # else :
    #     df_test = df_test_creator(df_all, units_index_test)
    #     print(df_test)
    #     print(df_test.columns)
    #     print("num of inputs: ", len(df_test.columns))
    #     df_train = pd.DataFrame()


    df_train = df_train_creator(df_all, units_index_train)
    print(df_train)
    print(df_train.columns)
    print("num of inputs: ", len(df_train.columns) )
    df_test = df_test_creator(df_all, units_index_test)
    print(df_test)
    print(df_test.columns)
    print("num of inputs: ", len(df_test.columns))

    del df_all
    gc.collect()
    df_all = pd.DataFrame()
    sample_dir_path = os.path.join(data_filedir, 'Samples_whole')
    sample_folder = os.path.isdir(sample_dir_path)
    if not sample_folder:
        os.makedirs(sample_dir_path)
        print("created folder : ", sample_dir_path)

    cols_normalize = df_train.columns.difference(['RUL', 'unit'])  # The function dataframe.columns.difference() gives you complement of the values that you provide as argument.
    sequence_cols = df_train.columns.difference(['RUL', 'unit'])  # It can be used to create a new dataframe from an existing dataframe with exclusion of some columns.


    if selector == 0:
        for unit_index in units_index_train:
            data_class = Input_Gen (df_train, df_test, cols_normalize, sequence_length, sequence_cols, sample_dir_path,
                                    unit_index, sampling, stride =stride)
            data_class.seq_gen()

    else:
        for unit_index in units_index_test:
            data_class = Input_Gen (df_train, df_test, cols_normalize, sequence_length, sequence_cols, sample_dir_path,
                                    unit_index, sampling, stride =stride)
            data_class.seq_gen()



if __name__ == '__main__':
    main()
    
    # python3 sample_creator_unit_auto.py -w 50 -s 1 --test 0 --sampling 10
    
'''
– w : window length
– s : stride of window
– test : select train or test, if it is zero, then the code extracts samples from the engines used for training. Otherwise, it creates samples from test engines
– sampling : subsampling the data before creating the output array so that we can set assume different sampling rate to mitigate memory issues.

Please note that we used N = 6 units (u = 2, 5, 10, 16, 18 & 20) for training and M = 3 units (u = 11, 14 & 15) for test, same as for the setting used in [3].

The size of the dataset is significantly large and it can cause memory issues by excessive memory use. 
Considering memory limitation that may occur when you load and create the samples, we set the data type as 'np.float32' to reduce the size of the data while the data type of the original data is 'np.float64'. Based on our experiments, this does not much affect to the performance when you use the data to train a DL network. If you want to change the type, please check 'data_preparation_unit.py' file in /utils folder.

In addition, we offer the data subsampling to handle 'out-of-memory' issues from the given dataset that use the sampling rate of 1Hz. 
When you set this subsampling input as 10, then it indicates you only take only 1 sample for every 10, the sampling rate is then 0.1Hz.

Finally, you can have 9 npz file in /N-CMAPSS/Samples_whole folder.

Each compressed file contains two arrays with different labels: 'sample' and 'label'. 
In the case of the test units, 'label' indicates the ground truth RUL of the test units for evaluation.

For instance, one of the created file, Unit2_win50_str1_smp10.npz, 
its filename indicates that the file consists of a collection of the sliced time series by time window size 50 from the trajectory of engine (unit) 2 with the sampling rate of 0.1Hz.

[3] Chao, Manuel Arias, Chetan Kulkarni, Kai Goebel, and Olga Fink. "Fusing physics-based and deep learning models for prognostics." Reliability Engineering & System Safety 217 (2022): 107961.
'''