# Task 1: Smile or NoSmile
#==============================================================================
# Import other python functions
import preprocess as prep
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

print("Task 1: Smile or NoSmile")

# Based on dataset of 5000 images
#train_num = 3750
#val_num = 250
#test_num = 1000
#train_batch_size = 32
test_split = 0.2 # 0.2 yields 3652/913 split

# Obtain filtered data
data = prep.remove_noise()
#print(data) #DEBUG

# Obtain training and testing datasets
train_df, test_df, train_f, test_f = prep.split_images_binary(data, test_split)
#print(train_f)
#print(test)
