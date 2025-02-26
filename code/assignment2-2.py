"""
Code for the second exercise of assignment 2 (8P361)
Submitted by Group 3
Instructions of Use: Run only
Additional packages: None
"""


# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

# import required packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard

def train_model(hidden_layers, activation_type):
    '''Creates, trains and tests a model with the specified amount of
    layers and activation type. Based on the code in mlp.py
    '''
    # preprocess
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=101)
    y_train, y_val = train_test_split(y_train, test_size=0.10, random_state=101)
    X_train = np.reshape(X_train, (-1,28,28,1))
    X_val = np.reshape(X_val, (-1,28,28,1))
    X_test = np.reshape(X_test, (-1,28,28,1))
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_val /= 255
    X_test /= 255
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)

    # Create Model
    model = Sequential()
    model.add(Flatten(input_shape=(28,28,1)))               # first layer
    for layer in range(hidden_layers):                      # hidden layers
        model.add(Dense(64, activation=activation_type))    # - variable amount and type!
    model.add(Dense(10, activation='softmax'))              # output layer

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model_name=f"model_hl{hidden_layers}_type{activation_type}"
    tensorboard = TensorBoard("logs/" + model_name)

    # train the model
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=[tensorboard])
    score = model.evaluate(X_test, y_test, verbose=0)

    print(f"Model with {hidden_layers} hidden {activation_type} layers:")
    print("Loss: ",score[0])
    print("Accuracy: ",score[1])

train_model(0,None) # No hidden layers
train_model(3,'relu') # 3 ReLu layers
train_model(3,'linear') # 3 Linear layers
