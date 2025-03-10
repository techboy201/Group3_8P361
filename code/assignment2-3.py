"""
Code for the third exercise of assignment 2 (8P361)
Submitted by Group 3
Instructions of Use: Run only
Additional packages: None
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard


# load the dataset using the builtin Keras method
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Define the categories
category_labels = {
    1: 0, 7: 0,  # 'vertical_digits'
    0: 1, 6: 1, 8: 1, 9: 1,  # 'loopy_digits'
    2: 2, 5: 2,  # 'curly_digits'
    3: 3, 4: 3   # 'other'
    }

# Convert y_train and y_test labels to new category labels
y_train = np.array([category_labels[label] for label in y_train])
y_test = np.array([category_labels[label] for label in y_test])

# derive a validation set from the training set
# the original training set is split into 
# new training set (90%) and a validation set (10%)
y_train_full = y_train
X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=101)
y_train, y_val = train_test_split(y_train, test_size=0.10, random_state=101)

def plt_classes(y, num_class=4):
    plt.figure()
    plt.hist(y, bins=range(0,num_class+1), align='left', rwidth=0.9)
    plt.xlabel('Class')
    plt.ylabel('Class count')
    plt.xticks([0,1,2,3],['vertical digits','loopy digits','curly digits','other'])
    plt.title('Class distribution')

# show the class label distribution in the training and validation dataset
plt_classes(y_train_full)
plt.show()

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

# class label preprocessing for keras
# convert 1D class arrays to 10D class matrices
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val,10)
y_test = to_categorical(y_test, 10)

model = Sequential()
# flatten the 28x28x1 pixel input images to a row of pixels (a 1D-array)
model.add(Flatten(input_shape=(28,28,1))) 
# fully connected layer with 64 neurons and ReLU nonlinearity
for i in range(5):
    model.add(Dense(256, activation='relu'))
# output layer with 10 nodes (one for each class) and softmax nonlinearity
model.add(Dense(10, activation='softmax')) 

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# use this variable to name your model
model_name="my_first_model"

# create a way to monitor our model in Tensorboard
tensorboard = TensorBoard("logs/{}".format(model_name))

# train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=[tensorboard])

score = model.evaluate(X_test, y_test, verbose=0)

print("Loss: ",score[0])
print("Accuracy: ",score[1])