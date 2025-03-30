import os 
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

# define path and get images
training_data_1 = '../datasets/train+val/train/1/'
training_data_0 = '../datasets/train+val/train/0/'
images_1 = os.listdir(training_data_1)
images_0 = os.listdir(training_data_0)

# show 5 random images per class
fig = plt.figure(figsize=(8, 4))
plt.title('Random training data images',fontweight='bold')
plt.axis('off')

for i in range(10):
    if i >= 5:
        img = training_data_1+random.choice(images_1)   # repeat for class = 1
        img_class = '1'
    else:
        img = training_data_0+random.choice(images_0)   # read random file
        img_class = '0'

    # read and plot plt
    img_data = cv2.imread(img)
    fig.add_subplot(2,5, i+1)
    plt.imshow(img_data)

    # format
    plt.axis('off')
    plt.title(img_class)
plt.show()