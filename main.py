# Task 1: Smile or NoSmile
#==============================================================================
# Import other python functions
import preprocess as prep
import classification as cls
# Import required libraries
import os
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
#from tensorflow import keras
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#from tensorflow.contrib import rnn
#from tensorflow.python.ops import variable_scope
#from tensorflow.python.framework import dtypes
#import copy

train_dest = prep.train_dest
test_dest = prep.test_dest

# Based on dataset of 5000 images
#train_num = 3750
#val_num = 250
#test_num = 1000
#train_batch_size = 32
test_split = 0.2 # 0.2 yields 3652/913 split
# smiling, eyeglasess
label_col = "human"

def weights_set(label_col):
    weights = {
        "eyeglasses": {0: 2.438, 1: 1.},
        "smiling": {0: 1., 1: 3.903},
        "young": {0: 1., 1: 3.8},
        "human": {0: 1.283, 1: 1.}
    }

# Obtain filtered data
data = prep.remove_noise()
#print(data) #DEBUG

# Obtain training and testing datasets
train_df, test_df, train_f, test_f = prep.split_images_binary(data, test_split)
test_df.to_csv("answers_" + label_col + ".csv", index = False)

cls.train_CNN_binary(train_df, test_df, label_col, train_dest, test_dest, weights_set(label_col))
