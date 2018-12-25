# Import required libraries
from keras.models import Sequential, Model #Initialise NN as sequence of layers
from keras.layers import Conv2D #For convoluting images (2D arrays)
from keras.layers import MaxPooling2D #To use Maxpooling function
from keras.layers import Flatten #To convert 2D arrays to a single linear vect
from keras.layers import Dense #To perform NN connections
from keras.layers import Dropout #For regularization
import keras.callbacks as callbacks #To use TensorBoard
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator #To augment images
from keras.utils import plot_model
from PIL import Image
import numpy as np
import pandas as pd
from keras import metrics
from sklearn.utils import class_weight
import sklearn.metrics as skmetrics

# Obtain training and testing directory paths
def get_paths(train_directory, test_directory):
    global train_dest
    global test_dest

    train_dest = train_directory
    test_dest = test_directory

    return 0

# A CNN trained to produce binary outputs
# Takes training and testing dataframes and their respective image directories plus class weights
# Trains, evaluates and tests a CNN as well as produce plots and a results csv file
def train_CNN(train_df, test_df, label_col, batch, weights, mode):
    #Top 3 categorical accuracy method
    def top_3_categorical_accuracy(y_true, y_pred):
        return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

    #Top 2 categorical accuracy method
    def top_2_categorical_accuracy(y_true, y_pred):
        return metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

    #Adam optimizer
    #Default: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False
    adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    #Check if binary of multiclass classification
    if mode == 0:
        output = 1
        final_activation = 'sigmoid'
        optimizer_mode = adam
        loss_mode = 'binary_crossentropy'
        metrics_mode = ['binary_accuracy']
        flow_mode = "binary"
        classes = [-1, 1]
    else:
        output = 7
        final_activation = 'softmax'
        optimizer_mode = 'rmsprop'
        loss_mode = 'categorical_crossentropy'
        metrics_mode = ['categorical_accuracy', top_2_categorical_accuracy, top_3_categorical_accuracy]
        flow_mode = "categorical"
        classes = [-1, 0, 1, 2, 3, 4, 5]

    classifier = Sequential() #Create Sequential object

    # Convolution layer with Non Linearity
    # To extract features from input
    # Filters: 32, Filter shape: 3X3, Input shape: 256X256, RGB(3), Activation function: Rectifier Linear Unit
    classifier.add(Conv2D(32, (3,3), input_shape = (256, 256, 3), activation = 'relu'))
    # Pooling layer using Max Pooling
    # To reduce the size of images and total number of nodes
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Dropout for regularisation
    #classifier.add(Dropout(0.25))

    #More convolution and pooling layers
    classifier.add(Conv2D(32, (3,3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    #classifier.add(Dropout(0.25))

    classifier.add(Conv2D(64, (3,3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    #classifier.add(Dropout(0.25))

    # Flattening
    # Convert 3D feature map to 1D feature vector
    classifier.add(Flatten())

    # Fully Connected/ Hidden layer
    # A fully connected layer
    # Nodes: 128, Activation function: Rectifier Linear Unit
    classifier.add(Dense(units = 128, activation = 'relu'))
    #classifier.add(Dropout(0.5))

    # Output layer
    # One node for binary output
    # Activation function: Sigmoid
    classifier.add(Dense(units = output, activation = final_activation))

    # Compile CNN
    # Optimiser parameter: adam, Loss function: Binary Crossentropy, Performance metric: Binary Accuracy
    classifier.compile(optimizer = optimizer_mode, loss = loss_mode, metrics = metrics_mode)

    #Realtime image augmentation
    #Rescale: rescales RGB values to 0-1
    #, rotation_range = 40, width_shift_range=0.2, height_shift_range=0.2, fill_mode='nearest'
    train_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.25, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255) #We don't modify the testing images as we are only doing predictions

    #Takes dataframe input and images directory and generate batches of augmented data
    train_generator = train_datagen.flow_from_dataframe(dataframe = train_df, directory = train_dest, x_col = "file_name", y_col = label_col, has_ext = False, batch_size = batch, seed = 42, shuffle = True, class_mode = flow_mode, target_size = (256, 256), subset = "training", classes = classes)
    valid_generator = train_datagen.flow_from_dataframe(dataframe = train_df, directory = train_dest, x_col = "file_name", y_col = label_col, has_ext = False, batch_size = batch, seed = 42, shuffle = True, class_mode = flow_mode, target_size = (256, 256), subset = "validation", classes = classes)
    test_generator = test_datagen.flow_from_dataframe(dataframe = test_df, directory = test_dest, x_col = "file_name", y_col = None, has_ext = False, batch_size = 1, seed = 42, shuffle = False, class_mode = None, target_size = (256, 256)) #Returns images

    #Obtain steps required for training, evaluating and testing
    step_size_train = train_generator.n//train_generator.batch_size
    step_size_valid = valid_generator.n//valid_generator.batch_size
    step_size_test = test_generator.n//test_generator.batch_size

    #Callbacks
    tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq = 0, write_graph = True, write_images = True)
    csv_logger = callbacks.CSVLogger('./logs/' + label_col + '_training_log.csv', separator=',', append=False)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=40, verbose=0, mode='auto')

    # Checks how to set class_weight parameter in fit_generator based on user input
    if weights == 'auto':
        index = train_generator.class_indices
        calc_class = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)
        class_weights = dict(zip(index.values(), calc_class))

    elif weights == 'nan_suppress':
        index = train_generator.class_indices
        calc_class = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)
        calc_class[0] = 0.
        class_weights = dict(zip(index.values(), calc_class))
    elif weights == 'equal':
        if mode == 1:
            class_weights = {0: 1., 1: 1., 2: 1., 3: 1., 4: 1., 5: 1., 6: 1.}
        else:
            class_weights = {0: 1., 1: 1.}
    else:
        class_weights = weights

    #Train model on generated data via batches
    train_history = classifier.fit_generator(train_generator, epochs = 100, class_weight = class_weights, steps_per_epoch = step_size_train, validation_data = valid_generator, validation_steps = step_size_valid, verbose = 1, callbacks = [csv_logger, early_stopping], workers = 10)

    classifier.summary()

    # Save weights
    classifier.save_weights('CNN.h5')

    # Evaluate model and assign loss and accuracy scores
    score = classifier.evaluate_generator(generator = valid_generator, steps = step_size_valid, verbose = 1)

    # Predict output
    test_generator.reset()
    prediction = classifier.predict_generator(test_generator, steps = step_size_test, verbose = 1) #Returns numpy array

    #Obtain predicted labels in an array through binary threshold or argmax
    if mode == 0:
        predicted_class_indices = [1 if x >= 0.5 else 0 for x in prediction]
    else:
        predicted_class_indices = np.argmax(prediction, axis = 1) #argmax across rows

    #Map predicted labels to their unique filenames
    labels = (train_generator.class_indices) #Get the class label indices (so we know what 0 and 1 refers to)
    labels = dict((v,k) for k,v in labels.items()) #Reverse indices
    predictions = [labels[k] for k in predicted_class_indices] #Get all predictions (class) from dict
    filenames = test_generator.filenames #Get all filenames of predictions
    results = pd.DataFrame({"Filename":filenames, "Predictions":predictions}) #Save filename and predictions to a dataframe

    return results, train_history, score

def train_MLP(train_df, test_df, label_col, batch, weights, mode):
    #Top 3 categorical accuracy method
    def top_3_categorical_accuracy(y_true, y_pred):
        return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

    #Top 2 categorical accuracy method
    def top_2_categorical_accuracy(y_true, y_pred):
        return metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

    #Adam optimizer
    #Default: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False
    adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    #Check if binary of multiclass classification
    if mode == 0:
        output = 1
        final_activation = 'sigmoid'
        optimizer_mode = adam
        loss_mode = 'binary_crossentropy'
        metrics_mode = ['binary_accuracy']
        flow_mode = "binary"
        classes = [-1, 1]
    else:
        output = 7
        final_activation = 'softmax'
        optimizer_mode = 'rmsprop'
        loss_mode = 'categorical_crossentropy'
        metrics_mode = ['categorical_accuracy', top_2_categorical_accuracy, top_3_categorical_accuracy]
        flow_mode = "categorical"
        classes = [-1, 0, 1, 2, 3, 4, 5]

    classifier = Sequential() #Create Sequential object

    # Fully Connected/ Hidden layer
    # A fully connected layer
    # Nodes: 128, Activation function: Rectifier Linear Unit
    classifier.add(Dense(units = 32, input_shape = (256, 256, 3), activation = 'relu'))
    #classifier.add(Dropout(0.5))

    classifier.add(Dense(units = 32, activation = 'relu'))

    classifier.add(Dense(units = 32, activation = 'relu'))

    # Flattening
    # Convert 3D feature map to 1D feature vector
    classifier.add(Flatten())

    # Output layer
    # One node for binary output
    # Activation function: Sigmoid
    classifier.add(Dense(units = output, activation = final_activation))

    # Compile CNN
    # Optimiser parameter: adam, Loss function: Binary Crossentropy, Performance metric: Binary Accuracy
    classifier.compile(optimizer = optimizer_mode, loss = loss_mode, metrics = metrics_mode)

    #Realtime image augmentation
    #Rescale: rescales RGB values to 0-1
    #, rotation_range = 40, width_shift_range=0.2, height_shift_range=0.2, fill_mode='nearest'
    train_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.25, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255) #We don't modify the testing images as we are only doing predictions

    #Takes dataframe input and images directory and generate batches of augmented data
    train_generator = train_datagen.flow_from_dataframe(dataframe = train_df, directory = train_dest, x_col = "file_name", y_col = label_col, has_ext = False, batch_size = batch, seed = 42, shuffle = True, class_mode = flow_mode, target_size = (256, 256), subset = "training", classes = classes)
    valid_generator = train_datagen.flow_from_dataframe(dataframe = train_df, directory = train_dest, x_col = "file_name", y_col = label_col, has_ext = False, batch_size = batch, seed = 42, shuffle = True, class_mode = flow_mode, target_size = (256, 256), subset = "validation", classes = classes)
    test_generator = test_datagen.flow_from_dataframe(dataframe = test_df, directory = test_dest, x_col = "file_name", y_col = None, has_ext = False, batch_size = 1, seed = 42, shuffle = False, class_mode = None, target_size = (256, 256)) #Returns images

    #Obtain steps required for training, evaluating and testing
    step_size_train = train_generator.n//train_generator.batch_size
    step_size_valid = valid_generator.n//valid_generator.batch_size
    step_size_test = test_generator.n//test_generator.batch_size

    #Callbacks
    tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq = 0, write_graph = True, write_images = True)
    csv_logger = callbacks.CSVLogger('./logs/' + label_col + '_training_log.csv', separator=',', append=False)

    # Checks how to set class_weight parameter in fit_generator based on user input
    if weights == 'auto':
        index = train_generator.class_indices
        calc_class = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)
        class_weights = dict(zip(index.values(), calc_class))

    elif weights == 'nan_suppress':
        index = train_generator.class_indices
        calc_class = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)
        calc_class[0] = 0.
        class_weights = dict(zip(index.values(), calc_class))
    elif weights == 'equal':
        if mode == 1:
            class_weights = {0: 1., 1: 1., 2: 1., 3: 1., 4: 1., 5: 1., 6: 1.}
        else:
            class_weights = {0: 1., 1: 1.}
    else:
        class_weights = weights

    #Train model on generated data via batches
    train_history = classifier.fit_generator(train_generator, epochs = 40, class_weight = class_weights, steps_per_epoch = step_size_train, validation_data = valid_generator, validation_steps = step_size_valid, verbose = 1, callbacks = [csv_logger], workers = 10)

    classifier.summary()

    # Save weights
    classifier.save_weights('MLP.h5')

    # Evaluate model and assign loss and accuracy scores
    score = classifier.evaluate_generator(generator = valid_generator, steps = step_size_valid, verbose = 1)

    # Predict output
    test_generator.reset()
    prediction = classifier.predict_generator(test_generator, steps = step_size_test, verbose = 1) #Returns numpy array

    #Obtain predicted labels in an array through binary threshold or argmax
    if mode == 0:
        predicted_class_indices = [1 if x >= 0.5 else 0 for x in prediction]
    else:
        predicted_class_indices = np.argmax(prediction, axis = 1) #argmax across rows

    #Map predicted labels to their unique filenames
    labels = (train_generator.class_indices) #Get the class label indices (so we know what 0 and 1 refers to)
    labels = dict((v,k) for k,v in labels.items()) #Reverse indices
    predictions = [labels[k] for k in predicted_class_indices] #Get all predictions (class) from dict
    filenames = test_generator.filenames #Get all filenames of predictions
    results = pd.DataFrame({"Filename":filenames, "Predictions":predictions}) #Save filename and predictions to a dataframe

    return results, train_history, score

def train_MultiVarCNN(train_df, test_df, label_col, batch, weights, mode):
    #Top 3 categorical accuracy method
    def top_3_categorical_accuracy(y_true, y_pred):
        return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

    #Top 2 categorical accuracy method
    def top_2_categorical_accuracy(y_true, y_pred):
        return metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

    #Adam optimizer
    #Default: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False
    adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    #Check if binary of multiclass classification
    if mode == 0:
        output = 1
        final_activation = 'sigmoid'
        optimizer_mode = adam
        loss_mode = 'binary_crossentropy'
        metrics_mode = ['binary_accuracy']
        flow_mode = "binary"
        classes = [-1, 1]
    else:
        output = 7
        final_activation = 'softmax'
        optimizer_mode = 'rmsprop'
        loss_mode = 'categorical_crossentropy'
        metrics_mode = ['categorical_accuracy', top_2_categorical_accuracy, top_3_categorical_accuracy]
        flow_mode = "categorical"
        classes = [-1, 0, 1, 2, 3, 4, 5]

    classifier = Sequential() #Create Sequential object

    # Convolution layer with Non Linearity
    # To extract features from input
    # Filters: 32, Filter shape: 3X3, Input shape: 256X256, RGB(3), Activation function: Rectifier Linear Unit
    classifier.add(Conv2D(32, (3,3), input_shape = (256, 256, 3), activation = 'relu'))
    # Pooling layer using Max Pooling
    # To reduce the size of images and total number of nodes
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Dropout for regularisation
    #classifier.add(Dropout(0.25))

    #More convolution and pooling layers
    classifier.add(Conv2D(32, (3,3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    #classifier.add(Dropout(0.25))

    classifier.add(Conv2D(64, (3,3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    #classifier.add(Dropout(0.25))

    # Flattening
    # Convert 3D feature map to 1D feature vector
    classifier.add(Flatten())

    # Fully Connected/ Hidden layer
    # A fully connected layer
    # Nodes: 128, Activation function: Rectifier Linear Unit
    classifier.add(Dense(units = 128, activation = 'relu'))
    #classifier.add(Dropout(0.5))

    # Output layer
    # One node for binary output
    # Activation function: Sigmoid
    classifier.add(Dense(units = output, activation = final_activation))

    # Compile CNN
    # Optimiser parameter: adam, Loss function: Binary Crossentropy, Performance metric: Binary Accuracy
    classifier.compile(optimizer = optimizer_mode, loss = loss_mode, metrics = metrics_mode)

    #Realtime image augmentation
    #Rescale: rescales RGB values to 0-1
    #, rotation_range = 40, width_shift_range=0.2, height_shift_range=0.2, fill_mode='nearest'
    train_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.25, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255) #We don't modify the testing images as we are only doing predictions

    #Takes dataframe input and images directory and generate batches of augmented data
    train_generator = train_datagen.flow_from_dataframe(dataframe = train_df, directory = train_dest, x_col = "file_name", y_col = "smiling", has_ext = False, batch_size = batch, seed = 42, shuffle = True, class_mode = flow_mode, target_size = (256, 256), subset = "training", classes = classes)
    valid_generator = train_datagen.flow_from_dataframe(dataframe = train_df, directory = train_dest, x_col = "file_name", y_col = label_col, has_ext = False, batch_size = batch, seed = 42, shuffle = True, class_mode = flow_mode, target_size = (256, 256), subset = "validation", classes = classes)
    test_generator = test_datagen.flow_from_dataframe(dataframe = test_df, directory = test_dest, x_col = "file_name", y_col = None, has_ext = False, batch_size = 1, seed = 42, shuffle = False, class_mode = None, target_size = (256, 256)) #Returns images


    #Obtain steps required for training, evaluating and testing
    step_size_train = train_generator.n//train_generator.batch_size
    step_size_valid = valid_generator.n//valid_generator.batch_size
    step_size_test = test_generator.n//test_generator.batch_size

    #Callbacks
    tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq = 0, write_graph = True, write_images = True)
    csv_logger = callbacks.CSVLogger('./logs/' + label_col + '_training_log.csv', separator=',', append=False)

    # Checks how to set class_weight parameter in fit_generator based on user input
    if weights == 'auto':
        index = train_generator.class_indices
        calc_class = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)
        class_weights = dict(zip(index.values(), calc_class))

    elif weights == 'nan_suppress':
        index = train_generator.class_indices
        calc_class = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)
        calc_class[0] = 0.
        class_weights = dict(zip(index.values(), calc_class))
    elif weights == 'equal':
        if mode == 1:
            class_weights = {0: 1., 1: 1., 2: 1., 3: 1., 4: 1., 5: 1., 6: 1.}
        else:
            class_weights = {0: 1., 1: 1.}
    else:
        class_weights = weights

    #Train model on generated data via batches
    train_history = classifier.fit_generator(train_generator, epochs = 40, class_weight = class_weights, steps_per_epoch = step_size_train, validation_data = valid_generator, validation_steps = step_size_valid, verbose = 1, callbacks = [csv_logger], workers = 10)

    classifier.summary()

    # Save weights
    classifier.save_weights('CNN.h5')

    # Evaluate model and assign loss and accuracy scores
    score = classifier.evaluate_generator(generator = valid_generator, steps = step_size_valid, verbose = 1)

    # Predict output
    test_generator.reset()
    prediction = classifier.predict_generator(test_generator, steps = step_size_test, verbose = 1) #Returns numpy array

    #Obtain predicted labels in an array through binary threshold or argmax
    if mode == 0:
        predicted_class_indices = [1 if x >= 0.5 else 0 for x in prediction]
    else:
        predicted_class_indices = np.argmax(prediction, axis = 1) #argmax across rows

    #Map predicted labels to their unique filenames
    labels = (train_generator.class_indices) #Get the class label indices (so we know what 0 and 1 refers to)
    labels = dict((v,k) for k,v in labels.items()) #Reverse indices
    predictions = [labels[k] for k in predicted_class_indices] #Get all predictions (class) from dict
    filenames = test_generator.filenames #Get all filenames of predictions
    results = pd.DataFrame({"Filename":filenames, "Predictions":predictions}) #Save filename and predictions to a dataframe

    return results, train_history, score
