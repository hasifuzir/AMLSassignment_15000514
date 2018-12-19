import matplotlib.pyplot as plt #For plots
import seaborn as sns #For nicer plots
import pandas as pd
import os

# Obtain results directory and create it if it doesn't exist
def get_paths(results_directory):
    global results_dir
    results_dir = results_directory

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print("Created Results folder!")

    return 0

# Inputs keras History object and plots metrics
def plot_graph(train_history, label_col, mode):

    # Obtain scores from history
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']

    if mode == 0:
        acc = train_history.history['binary_accuracy']
        val_acc = train_history.history['val_binary_accuracy']
    else:
        acc = train_history.history['categorical_accuracy']
        val_acc = train_history.history['val_categorical_accuracy']

    # Plot loss scores
    sns.set_style("whitegrid")
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model Loss')
    plt.legend(['Loss', 'Validation Loss'])
    plt.savefig(results_dir + '/' + label_col + '_loss.png')
    plt.show()

    # Plot accuracy scores
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Model Accuracy')
    plt.legend(['Accuracy', 'Validation Accuracy'])
    plt.savefig(results_dir + '/' + label_col + '_acc.png')
    plt.show()

    return 0

# Save prediction results to csv
def save_pred(results, label_col):
    results.to_csv(results_dir + "/results_" + label_col + ".csv", header = False, index = False)
    print("Saved results file to results_" + label_col + ".csv !")

    return 0
