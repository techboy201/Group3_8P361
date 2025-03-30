"""
Code for the first exercise of assignment 2 (8P361)
Submitted by Group 3
Instructions of Use: Run only
Additional packages: keras-tuner
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import itertools
import tensorflow as tf
import pandas as pd
import keras_tuner as kt # pip install keras-tuner==1.3.5

# import required packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard

print(f'using tensorflow {tf.version.VERSION} with GPU:')
print(tf.config.list_physical_devices('GPU'))

# load the dataset using the builtin Keras method
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# derive a validation set from the training set
# the original training set is split into 
# new training set (90%) and a validation set (10%)
X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=101)
y_train, y_val = train_test_split(y_train, test_size=0.10, random_state=101)

# the shape of the data matrix is NxHxW, where
# N is the number of images,
# H and W are the height and width of the images
# keras expect the data to have shape NxHxWxC, where
# C is the channel dimension
X_train = np.reshape(X_train, (-1,28,28,1)) 
X_val = np.reshape(X_val, (-1,28,28,1))
X_test = np.reshape(X_test, (-1,28,28,1))

# convert the datatype to float32
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

# normalize our data values to the range [0,1]
X_train /= 255
X_val /= 255
X_test /= 255

# convert 1D class arrays to 10D class matrices
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)
y_test = to_categorical(y_test, 10)

# Function to build a model with hyperparameters
def build_model(hp):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))

    # Choose parameters
    num_layers = hp.Choice('num_layers', [1, 5, 10])
    num_neurons = hp.Choice('num_neurons', [64, 128, 256])

    # Add variable hidden layers
    for _ in range(num_layers):
        model.add(Dense(num_neurons, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

# Use KerasTuner for hyperparameter tuning
# when run once, is stored in hp_assignment_2 and not run again
tuner = kt.GridSearch(build_model,
                      objective='val_accuracy',
                      max_trials=9,  # 3 neuron choices * 3 layer choices
                      executions_per_trial=1,  # Run each trial once
                      project_name='hp_assignment_2')  # Change this to force a fresh tuning run

# Perform the search
tuner.search(X_train, y_train, epochs=3, batch_size=16, validation_data=(X_val, y_val), verbose=1)

# Get the best hyperparameters
hps = tuner.get_best_hyperparameters(num_trials=1)
best_hps = hps[0]

print(f"Best number of layers: {best_hps.get('num_layers')}")
print(f"Best number of neurons: {best_hps.get('num_neurons')}")

# Train the best model
best_model = tuner.hypermodel.build(best_hps)
best_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate on test set
score = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Best Model Loss: {score[0]}, Best Model Accuracy: {score[1]}")

# Print full results
for trial in tuner.oracle.trials.values():
    num_layers = trial.hyperparameters.values['num_layers']
    num_neurons = trial.hyperparameters.values['num_neurons']
    val_acc = trial.metrics.get_best_value('val_accuracy')  # Get the best validation accuracy
    val_loss = trial.metrics.get_best_value('val_loss')  # Get the best validation loss

    print(f"{num_layers} layers, {num_neurons} neurons - Val LOSS: {val_loss}, Val ACC: {val_acc}")
