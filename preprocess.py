# Import required libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import shutil

#Get base, images and label CSV paths from main
def get_paths(images_directory, train_directory, test_directory):
    global images_dir
    global train_dest
    global test_dest

    images_dir = images_directory
    train_dest = train_directory
    test_dest = test_directory

    return 0

# Removes noise images from the labels
def remove_noise(labels_file):
    # Read dataset
    df = pd.read_csv(labels_file, skiprows = 1) #Skip the initial row (classes)
    # Available features: file_name,hair_color,eyeglasses,smiling,young,human
    # Filer out noise (All noise has a value of -1 across all features)
    df_filtered = df.loc[~((df['hair_color'] == -1) & (df['eyeglasses'] == -1) & (df['smiling'] == -1) & (df['young'] == -1) & (df['human'] == -1))]

    return df_filtered

# Splits images dataset into train and validation directories
def split_images_binary(df, split):
    train, test = train_test_split(df, test_size = split)

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

    return train, test

# Save test answers to csv
def save_answers(test_df, label_col):
    test_df.to_csv("answers_" + label_col + ".csv", index = False)
    print("Saved answer file to answers_" + label_col + ".csv !")

    return 0
