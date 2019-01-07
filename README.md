# AMLSassignment: 15000514
Applied Machine Learning Systems ELEC0132 (18/19) Assignment for Student 15000514

Requires Python3 (3.6.7).

**Libraries required (version used)**
 - Tensorflow-gpu (1.12.0)
 - Keras (2.2.4)
 - Matplotlib (3.0.2)
 - Numpy (1.15.4)
 - OpenCV-python (3.4.4.19)
 - Pandas (0.23.4)
 - Pillow (5.3.0)
 - Scikit-learn (0.20.1)
 - Seaborn (0.9.0)

Use Pip to install required libraries:

    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
    pip install <LIBRARY>
Includes 3 models: CNN, MLP and Inception v3. Run ***main.py*** file.

## main.py
Main file to run.
Includes several parameters for user customisation.

| Parameter | Description  |
| --- | --- |
|  base_dir | Dataset directory |
| images_dir | Dataset folder within dataset directory |
| labels_file | Attribute list CSV within dataset directory |
| train_dest | Train images folder (will be created) |
| test_dest | Test images folder (will be created) |
| results_dir | Results directory (for prediction CSV and loss & accuracy graphs |
| batch_size | Size of batches for training/validation |
| test_split | Train-test split ratio (0.2 yields a 3652/913 split) |
| label_col | attribute to train and test on |
| weights_mode | Method to balance class weights |

*label_col* parameters:

 - smiling (task_1)
 - young (task_2)
 - eyeglasses (task_3)
 - human (task_4)
 - hair_color (task_5)

*weights_mode* parameters:

 - manual - Manually set weights in `manual_weights_set` method, customisable. Default values is based on dataset.
 - auto- Automatically calculate based on created train folder distributed using scikit-learn's `train_test_split` method
 - Equal - Treat all classes as equal
 - nan_suppress - Same as auto, except -1 (NaN) class weight is set to 0. Only for *hair_color* task.

Model methods (uncomment as necessary in file):

    cls.train_CNN
    cls.train_MLP
    cls.train_Inception

Running the file will train and produce predictions based on user input task. Training graphs are saved in `results` folder. Prediction CSV file is saved in `results` folder. Training logs are saved in `logs` folder.

## classification.py
Contains all classification model methods.
`Train_CNN` contains several customisable parameters:

| Parameter | Description  |
| --- | --- |
|  lr | Adam learning rate |
| output | Number of neurons in output layer |
| final_activation | Activation of output layer |
| optimizer_mode | Optimizer used |
| loss_mode | Loss function used |
| metrics_mode | Performance metric to observe |
| classes | Dict to list all class indices |
| units| Number of neurons |

`ImageDataGenerator` can be modified to customise image augmentations.

`callbacks.EarlyStopping` 's `patience` parameter can be modified.

`classifier.fit_generator` 's `epoch` parameter can be modified to set epochs. `callbacks` can be modified to include or remove any callback methods.


`Train_MLP` contains several customisable parameters:

| Parameter | Description  |
| --- | --- |
|  lr | Adam learning rate |
| output | Number of neurons in output layer |
| final_activation | Activation of output layer |
| optimizer_mode | Optimizer used |
| loss_mode | Loss function used |
| metrics_mode | Performance metric to observe |
| classes | Dict to list all class indices |
| units| Number of neurons |

`ImageDataGenerator` can be modified to customise image augmentations.

`callbacks.EarlyStopping` 's `patience` parameter can be modified.

`classifier.fit_generator` 's `epoch` parameter can be modified to set epochs. `callbacks` can be modified to include or remove any callback methods.


`Train_Inception` contains several customisable parameters:

| Parameter | Description  |
| --- | --- |
|  lr | Adam learning rate |
| output | Number of neurons in output layer |
| final_activation | Activation of output layer |
| optimizer_mode | Optimizer used |
| loss_mode | Loss function used |
| metrics_mode | Performance metric to observe |
| classes | Dict to list all class indices |

`ImageDataGenerator` can be modified to customise image augmentations.

`callbacks.EarlyStopping` 's `patience` parameter can be modified.

`classifier.fit_generator` 's `epoch` parameter can be modified to set epochs. `callbacks` can be modified to include or remove any callback methods.


## analysis.py
Contains methods to plot graphs and save predictions.


## preprocess.py
Contains methods for preprocessing.

## How to perform predictions on separate dataset
Save images for predictions in a folder within folder in the `dataset` folder. Avoid `test` folder name! Keras' `flow_from_directory` method still requires test folder to have a subfolder!
E.g:

    ./dataset/prediction/images/<images>

Modify `test_dest` parameter in **main.py** to the test directory.
E.g:

    test_dest = ./dataset/prediction/

Uncomment desired model to train and predict from.

Modify `test_dest` parameter in `get_paths` method in classification.py method.
E.g:

    test_dest = "./dataset/prediction/"

Uncomment `test_generator` parameter to use `flow_from_directory` method. Comment the one using the `flow_from_dataframe` method.
E.g:

        test_generator = test_datagen.flow_from_directory(directory = test_dest, batch_size = 1, seed = 42, shuffle = False, class_mode = None, target_size = (256, 256)) #Returns images

Run **main.py**. Chosen model will train and perform predictions on the new test images. Predictions are saved as a CSV file in the `results` folder as `task_<TASK NUMBER>.csv`
