#==============================================================================
# Import required modules
import preprocess as prep
import classification as cls
import analysis
# Import required libraries
import os

# Set paths
base_dir = './dataset'
images_dir = os.path.join(base_dir,'images')
labels_file = os.path.join(base_dir,'attribute_list.csv')
train_dest = os.path.join(base_dir,'train')
test_dest = os.path.join(base_dir,'test')
results_dir = "./results"

# Set parameters
# Based on dataset of 5000 images
batch_size = 32 #Size of batches for training and validation
test_split = 0.2 #Ratio to split into training and testing images dataset, 0.2 yields 3652/913 split
label_col = "eyeglasses" #Class to train, validate and test on

# Set custom class weights to compensate for imbalanced data
def weights_set(label_col):
    weights = {
        # weight_index: dataset_index
        # Binary classes
        # 0: -1, 1: 1
        "eyeglasses": {0: 1., 1: 2.4375},
        "smiling": {0: 3.90333, 1: 1.},
        "young": {0: 3.80021, 1: 1.},
        "human": {0: 1., 1: 1.2825},

        # Multiclass classes
        # 0: -1, 1: 0, 2: 1, 3: 2, 4: 3, 5: 5, 6: 5
        "hair_color": {0: 4.364035, 1: 11.30682, 2: 1., 3: 1.819013, 4: 1.055143, 5: 1.26269, 6: 1.839187}
    }

# Pass paths to modules that require them
prep.get_paths(images_dir, train_dest, test_dest)
cls.get_paths(train_dest, test_dest)
analysis.get_paths(results_dir)

# Set whether binary or multiclass prediction
if label_col == "hair_color":
    pred_mode = 1
else:
    pred_mode = 0

# Obtain filtered data
data = prep.remove_noise(labels_file)

# Obtain training and testing datasets and save testing answers as csv file
train_df, test_df = prep.split_images_binary(data, test_split)
prep.save_answers(test_df, label_col)

# Train CNN and obtain results csv, keras training history and metric scores
results, train_history, score = cls.train_CNN(train_df, test_df, label_col, batch_size, weights_set(label_col), pred_mode)

# Print Test loss and accuracy
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plot graphs based on loss and accuracy for training and validation
analysis.plot_graph(train_history, label_col, pred_mode)

# Save predriction results to csv
analysis.save_pred(results, label_col)

#Pass prediction dataframes so we can plot graphs and stuff?
