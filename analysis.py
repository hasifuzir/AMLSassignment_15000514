import matplotlib.pyplot as plt #For plots
import matplotlib.axes as ax
import seaborn as sns #For nicer plots
import pandas as pd
import os
import csv

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
    loss = train_history.history['loss'] #List
    val_loss = train_history.history['val_loss']


    if mode == 0:
        acc = train_history.history['binary_accuracy']
        val_acc = train_history.history['val_binary_accuracy']
    else:
        acc = train_history.history['categorical_accuracy']
        val_acc = train_history.history['val_categorical_accuracy']

    # Plot loss scores
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 1)
    ax.plot(loss, label = "Loss")
    ax.plot(val_loss, label = "Validation Loss")
    ax.set_title('Model Loss')
    ax.legend(loc = "upper right")
    ax.set_xlim([0, 50])
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    ax.minorticks_on()
    ax.grid(b=True, which='major')
    ax.grid(b=True, which='minor')
    plt.savefig(results_dir + '/' + label_col + '_loss.png')
    plt.show()

    # Plot accuracy scores
    fig, ax = plt.subplots(1, 1)
    ax.plot(acc, label = "Accuracy")
    ax.plot(val_acc, label = "Validation Accuracy")
    ax.set_title('Model Accuracy')
    ax.legend(loc = "lower right")
    ax.set_xlim([0, 50])
    ax.grid(b=True, which='major')
    ax.grid(b=True, which='minor')
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epochs")
    ax.minorticks_on()
    plt.savefig(results_dir + '/' + label_col + '_acc.png')
    plt.show()

    return 0

# Save prediction results to csv
def save_pred(results, label_col, score):
    def task_set(label_col):
        return {
            # weight_index: dataset_index
            # Binary classes
            # 0: -1, 1: 1
            "smiling": "1",
            "young": "2",
            "eyeglasses": "3",
            "human": "4",
            "hair_color": "5"
        }.get(label_col, '0') #rturns proper class weight, else returns auto which will let sklearn to calculate it automatically

    with open(results_dir + "/task_" + task_set(label_col) + ".csv", "w", newline = '') as csv_file:
        writer = csv.writer(csv_file, delimiter = ',')
        writer.writerow([round(score, 4), ''])

    results.to_csv(results_dir + "/task_" +  task_set(label_col) + ".csv", header = False, index = False, mode = 'a')

    print("Saved results file to task_" +  task_set(label_col) + ".csv !")

    return 0
