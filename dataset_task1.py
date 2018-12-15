# Task 1: Smile or NoSmile
#==============================================================================
# Import required libraries
import os
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
#from tensorflow.contrib import rnn
#from tensorflow.python.ops import variable_scope
#from tensorflow.python.framework import dtypes
#import tensorflow as tf
#import copy

print("Task 1: Smile or NoSmile")

# Set paths
global base_dir
base_dir = './dataset'
images_dir = os.path.join(base_dir,'images')
labels_file = os.path.join(base_dir,'attribute_list.csv')

#read dataset
df = pd.read_csv(labels_file, skiprows=1) #Skip the initial row

# Copy all columns from the dataset for preprocessing
# Available features: file_name,hair_color,eyeglasses,smiling,young,human
#raw = df.values.copy()
#print(raw.shape) #DEBUG
#print(raw) #DEBUG

#Filer out noise (All noise has a value of -1 across all features)
df_filtered = df.loc[(df['hair_color'] == -1) & (df['eyeglasses'] == -1) & (df['smiling'] == -1) & (df['young'] == -1) & (df['human'] == -1)]



#X_raw = df.loc[:, ['file_name', 'smiling']].values.copy()
