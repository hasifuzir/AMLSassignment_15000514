# Import required libraries
from keras.models import Sequential #Initialise NN as sequence of layers
from keras.layers import Conv2D #For convoluting images (2D arrays)
from keras.layers import MaxPooling2D #To use Maxpooling function
from keras.layers import Flatten #To convert 2D arrays to a single linear vect
from keras.layers import Dense #To perform NN connections
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

def train_CNN_binary(train_df, test_df, label_col, train_dest, test_dest, weights):
    classifier = Sequential() #Create Sequential object

    # Convolution
    # Filters: 32
    # Filter shape: 3X3
    # Input shape: 256X256, RGB(3)
    # Activation function: Rectifier
    classifier.add(Conv2D(32, (3,3), input_shape = (256, 256, 3), activation = 'relu'))

    # Pooling
    # To reduce the size of images and total number of nodes
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Conv2D(32, (3,3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Conv2D(64, (3,3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattening
    # Convert 3D feature map to 1D feature vector
    classifier.add(Flatten())

    # Hidden layer
    # Nodes: 128
    # Activation function: Rectifier
    classifier.add(Dense(units = 128, activation = 'relu'))

    # Output layer
    # One node for binary output
    # Activation function: Sigmoid
    classifier.add(Dense(units = 1, activation = 'sigmoid'))

    # Compile CNN
    # Optimiser parameter: adam
    # Loss function
    # Performance metric
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, validation_split = 0.25)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    train_generator = train_datagen.flow_from_dataframe(dataframe = train_df, directory = train_dest, x_col = "file_name", y_col = label_col, has_ext = False, batch_size = 32, seed = 42, shuffle = True, class_mode = "binary", target_size = (256, 256), subset = "training")
    valid_generator = train_datagen.flow_from_dataframe(dataframe = train_df, directory = train_dest, x_col = "file_name", y_col = label_col, has_ext = False, batch_size = 32, seed = 42, shuffle = True, class_mode = "binary", target_size = (256, 256), subset = "validation")
    test_generator = test_datagen.flow_from_dataframe(dataframe = test_df, directory = test_dest, x_col = "file_name", y_col = None, has_ext = False, batch_size = 1, seed = 42, shuffle = False, class_mode = None, target_size = (256, 256)) #Batch size needs to be divisible by 913

    step_size_train = train_generator.n//train_generator.batch_size
    step_size_valid = valid_generator.n//valid_generator.batch_size
    step_size_test = test_generator.n//test_generator.batch_size

    train_history = classifier.fit_generator(train_generator, steps_per_epoch = step_size_train, validation_data = valid_generator, validation_steps = step_size_valid, epochs = 10, class_weight = weights, verbose = 1)

    # Evaluate model
    score = classifier.evaluate_generator(generator = valid_generator, steps = step_size_valid, verbose = 1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    loss = train_history.history['loss']
    acc = train_history.history['binary_accuracy']
    val_loss = train_history.history['val_loss']
    val_acc = train_history.history['val_binary_accuracy']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Model accuracy')
    plt.legend(['loss', 'val_loss', 'binary_accuracy', 'val_binary_accuracy'])
    plt.show()

    # Predict output
    test_generator.reset()
    prediction = classifier.predict_generator(test_generator, steps = step_size_test, verbose = 1) #Returns numpy array

    #predicted_class_indices = np.argmax(prediction, axis = 1)
    predicted_class_indices = [1 if x >= 0.5 else 0 for x in prediction]
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    filenames = test_generator.filenames
    results=pd.DataFrame({"Filename":filenames, "Predictions":predictions})
    results.to_csv("results_" + label_col + ".csv", header = False, index = False)

    return 0
