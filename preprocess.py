# Import required libraries
import os
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import cv2
import shutil

# Set paths
base_dir = './dataset'
images_dir = os.path.join(base_dir,'images')
labels_file = os.path.join(base_dir,'attribute_list.csv')
test_dest = os.path.join(base_dir,'test')
train_dest = os.path.join(base_dir,'train')

# Removes noise images from the labels
def remove_noise():
    # Read dataset
    df = pd.read_csv(labels_file, skiprows=1) #Skip the initial row
    # Available features: file_name,hair_color,eyeglasses,smiling,young,human
    # Filer out noise (All noise has a value of -1 across all features)
    df_filtered = df.loc[~((df['hair_color'] == -1) & (df['eyeglasses'] == -1) & (df['smiling'] == -1) & (df['young'] == -1) & (df['human'] == -1))]

    return df_filtered

# Splits images dataset into train and validation directories
def split_images_binary(df, split):
    train, test = train_test_split(df, test_size = split)
    train_filename = train.loc[:, 'file_name']
    test_filename = test.loc[:, 'file_name']

    if os.path.exists(train_dest):
        shutil.rmtree(train_dest)
        print("Deleted previous Train folder!")
        os.mkdir(train_dest)
        print("Created Train folder!")

    else:
        os.mkdir(train_dest)
        print("Created Train folder!")

    if os.path.exists(test_dest):
        shutil.rmtree(test_dest)
        print("Deleted previous Test folder!")
        os.mkdir(test_dest)
        print("Created Test folder!")

    else:
        os.mkdir(test_dest)
        print("Created Test folder!")

    for index, rows in train.iterrows():
        img = cv2.imread(images_dir + "/" + str(rows['file_name']) + ".png")
        cv2.imwrite(train_dest + "/" + str(rows['file_name']) + ".png", img)

    print("Copied training images into Train folder!")

    for index, rows in test.iterrows():
        img = cv2.imread(images_dir + "/" + str(rows['file_name']) + ".png")
        cv2.imwrite(test_dest + "/" + str(rows['file_name']) + ".png", img)

    print("Copied testing images into Test folder!")

    return train, test, train_filename, test_filename

def extract_features():

    return 0
