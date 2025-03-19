from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2


def load_model(model_name, weights):
    # Load the model
    with open(model_name, "r") as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)

    # Load the model weights
    model.load_weights(weights)

    print("Model and weights successfully loaded.")
    return model


def load_image(image_path):
    # Due to the images being a '.tif' file, we need to use PIL
    image = Image.open(image_path)
    image = image.resize((96, 96))
    image = np.array(image) / 255.0  # Normalise
    image = np.expand_dims(image, axis=0)  # add batch dimension

    print("Image successfully loaded.")
    return image

def compute_gradcam(model, img_array, layer_name="conv2d_3"):
    """
    Generates a Grad-Cam heatmap for a given image and model.
    :param model: The previously loaded model
    :param img_array: The previously loaded image
    :param layer_name: Name of the convolutional layer in your model
    :return: The heatmap
    """
    grad_model = Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = np.mean(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

def overlay_gradcam(original_image, heatmap, alpha=0.4):
    """
    Overlay the Grad-CAM heatmap over the original image.
    :param original_image: The previously loaded image
    :param heatmap: The generated Grad-CAM heatmap
    :param alpha: Transparancy level of the heatmap
    :return: Merged image with overlay
    """

    # Convert PIL image (0-1 range) to 0-255 range for visualization
    image = (original_image.squeeze() * 255).astype(np.uint8)  # convert back to 8-bit

    # Scale the heatmap to 0-255 and convert to colormap (JET)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))  # Resize heatmap to imput image
    heatmap = np.uint8(255 * heatmap)  # Normalise to 0-255
    heatmap = cm.jet(heatmap)[:, :, :3]  # Matplotlib colormap (JET), deletes alpha channel
    heatmap = np.uint8(255 * heatmap)  # convert to 8-bit

    # Make an overlay using alpha blending
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return overlay


# Loading the model
model_name = "my_second_cnn_model.json"
model_weights = "my_second_cnn_model_weights.hdf5"
model = load_model(model_name,model_weights)

# Loading the image
image_path = r"D:\School\Project AI for MIA\data\test\0000ec92553fda4ce39889f9226ace43cae3364e.tif"
image = load_image(image_path)

# Generate Grad-CAM heatmap
heatmap = compute_gradcam(model, image, "conv2d_3")

# Plot the Grad-CAM heatmap
plt.matshow(heatmap, cmap='jet')
plt.colorbar()
plt.title("Grad-CAM Heatmap")
plt.show()

# Generate the overlay
overlay_image = overlay_gradcam(image.squeeze(), heatmap, alpha=0.5)

# present image with heatmap overlay
plt.figure(figsize=(6, 6))
plt.imshow(overlay_image)
plt.axis("off")
plt.title("Grad-CAM Overlay")
plt.show()
