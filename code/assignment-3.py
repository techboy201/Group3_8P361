'''
TU/e BME Project Imaging 2021
Convolutional neural network for PCAM
Author: Suzanne Wetstein
'''

# disable overly verbose tensorflow logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import model_from_json
# unused for now, to be used for ROC analysis
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt


# the size of the images in the PCAM dataset
IMAGE_SIZE = 96


def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

     # dataset parameters
     train_path = os.path.join(base_dir, 'train+val', 'train')
     valid_path = os.path.join(base_dir, 'train+val', 'valid')


     RESCALING_FACTOR = 1./255

     # instantiate data generators
     datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

     train_gen = datagen.flow_from_directory(train_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary')

     val_gen = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             class_mode='binary')

     return train_gen, val_gen


def get_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64):

     # build the model
     model = Sequential()

     model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Flatten())
     model.add(Dense(64, activation = 'relu'))
     model.add(Dense(1, activation = 'sigmoid'))


     # compile the model
     model.compile(SGD(learning_rate=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

     return model


# get the model
model = get_model()


# get the data generators
train_gen, val_gen = get_pcam_generators(r'C:\Users\20212208\OneDrive - TU Eindhoven\Desktop\8P361\Project_8P')



# save the model and weights
model_name = 'my_first_cnn_model'
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.hdf5'

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)


# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]


# train the model
train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size

history = model.fit(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=3,
                    callbacks=callbacks_list)

# ROC analysis

# Define paths
VALIDATION_DIR = r"C:\Users\20212208\OneDrive - TU Eindhoven\Desktop\8P361\Project_8P\train+val\valid"
MODEL_JSON_PATH = r"C:\Users\20212208\OneDrive - TU Eindhoven\Desktop\8P361\Project_8P\my_first_cnn_model.json"
MODEL_WEIGHTS_PATH = r"C:\Users\20212208\OneDrive - TU Eindhoven\Desktop\8P361\Project_8P\my_first_cnn_model_weights.hdf5"

# Load model architecture from JSON
with open(MODEL_JSON_PATH, "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)  # Recreate model from JSON structure
model.load_weights(MODEL_WEIGHTS_PATH)  # Load trained weights

# Load validation dataset
IMG_SIZE = (96, 96)  # Adjust to match original training size
BATCH_SIZE = 32

datagen = ImageDataGenerator(rescale=1.0/255)  # Normalize images
valid_generator = datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False  # Keep order for ROC computation
)

# Get ground truth labels
y_true = valid_generator.classes

# Generate predictions
y_pred_probs = model.predict(valid_generator)  # Probabilities
y_pred_labels = (y_pred_probs > 0.5).astype(int)  # Convert to binary labels

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# Convolutional layers only model

def get_model_conv(kernel_size=(3, 3), pool_size=(4, 4), first_filters=32, second_filters=64, third_filters=128):
    model = Sequential()

    model.add(Conv2D(first_filters, kernel_size, activation='relu', padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Conv2D(second_filters, kernel_size, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Conv2D(third_filters, kernel_size, activation='relu', padding='same'))  # Added third convolutional layer
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Changed to single neuron with sigmoid

    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.95), loss='binary_crossentropy',
                  metrics=['accuracy'])  # Changed loss function

    return model
