import os

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Flatten, Dense
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# unused for now, to be used for ROC analysis
from sklearn.metrics import roc_curve, auc

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

# Function to create a mixed generator
def mixed_generator(orig_gen, conf_gen, ratio):
    '''Custom generator to get a mix of original and confounded images.'''
    while True:
        orig_images, orig_labels = orig_gen.next()
        conf_images, conf_labels = conf_gen.next()
        
        # Number of confounded images to mix in
        conf_count = int(train_batch_size * ratio)
        orig_count = train_batch_size - conf_count
        
        # Select subsets of original and confounded data
        mixed_images = np.concatenate((orig_images[:orig_count], conf_images[:conf_count]), axis=0)
        mixed_labels = np.concatenate((orig_labels[:orig_count], conf_labels[:conf_count]), axis=0)
        
        # Ensure we don't request more indices than available
        num_samples = len(mixed_images)
        batch_size = min(train_batch_size, num_samples)  # Use smaller of batch size or available data

        # Randomly sample indices safely
        indices = np.random.choice(num_samples, batch_size, replace=False)  # Ensure valid indices

        # Shuffle the mixed data
        np.random.shuffle(indices)
        mixed_images, mixed_labels = mixed_images[indices], mixed_labels[indices]

        yield mixed_images, mixed_labels

def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32, confounder_percentage=0.0):
    # Paths for original and modified datasets
    TRAIN_PATH_ORIG = os.path.join(base_dir, 'train+val', 'train')
    TRAIN_PATH_CONF = os.path.join(base_dir, 'train+val', 'train', '1_modified')  # Confounded data

    VALID_PATH_ORIG = os.path.join(base_dir, 'train+val', 'valid')
    VALID_PATH_CONF = os.path.join(base_dir, 'train+val', 'valid', '1_modified')

    RESCALING_FACTOR = 1. / 255

    # instantiate data generators
    datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

    # Original training data generator
    train_gen_orig = datagen.flow_from_directory(TRAIN_PATH_ORIG,
                                                 target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                 batch_size=train_batch_size,
                                                 class_mode='binary')

    # Confounded training data generator
    train_gen_conf = datagen.flow_from_directory(TRAIN_PATH_CONF,
                                                 target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                 batch_size=train_batch_size,
                                                 class_mode='binary')

    # Original validation data generator
    val_gen_orig = datagen.flow_from_directory(VALID_PATH_ORIG,
                                               target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                               batch_size=val_batch_size,
                                               class_mode='binary')

    # Confounded validation data generator (if needed)
    val_gen_conf = datagen.flow_from_directory( VALID_PATH_CONF,
                                                target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                batch_size=val_batch_size,
                                                class_mode='binary')

    # Return mixed training generator and standard validation generator
    train_gen_mixed = mixed_generator(train_gen_orig, train_gen_conf, confounder_percentage)
    n_train_samples = train_gen_orig.n
    n_val_samples = val_gen_orig.n

    return train_gen_mixed, val_gen_orig, n_train_samples, n_val_samples


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

# confounded data percentage
confounder_percentage = 0.3  # 30% confounded data

# batch sizes
train_batch_size = 32
val_batch_size = 32

# modified data generators and sample counts
train_gen, val_gen, n_train_samples, n_val_samples = get_pcam_generators('../datasets', confounder_percentage=confounder_percentage)

for layer in model.layers:
    print(layer.output_shape)

# save the model and weights
model_name = f'cnn_model_{round(confounder_percentage*100)}%_confounded'
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.hdf5'

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json) 

# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]

# steps per epoch
train_steps = n_train_samples // train_batch_size
val_steps = n_val_samples // val_batch_size  # Now correctly referencing validation sample count

# train model
history = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=3,
    callbacks =callbacks_list
)