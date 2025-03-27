from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import os

def load_model(model_name, weights):
    # Load the model
    with open(model_name, "r") as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)

    # Load the model weights
    model.load_weights(weights)

    return model


def load_image(image_path):
    # Due to the images being a '.tif' file, we need to use PIL
    image = Image.open(image_path)
    image = image.resize((96, 96))
    image = np.array(image) / 255.0  # Normalise
    image = np.expand_dims(image, axis=0)  # add batch dimension

    return image

def compute_gradcam(model, img_array, layer_name="conv2d_2"):
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
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    else:
        heatmap[:] = 0

    return heatmap, predictions.numpy()[0][0]

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


def create_dot_mask(shape, x, y, radius):
    mask = np.zeros(shape, dtype=np.float32)
    cv2.circle(mask, (x, y), radius, 1, -1)
    return mask


def run_gmi_analysis(model_json, weights, csv_path, image_dir, conv_layer_name):
    model = load_model(model_json, weights)
    df = pd.read_csv(csv_path)

    results = []

    for i, row in df.iterrows():
        image_path = os.path.join(image_dir, row["image_name"])
        try:
            image = load_image(image_path)
            heatmap, pred = compute_gradcam(model, image, layer_name=conv_layer_name)

            # Get the coordinates and radius
            x, y, r = int(row["x"]), int(row["y"]), int(row["radius"])

            # Scale coordinates to the heatmap-resolution
            heatmap_h, heatmap_w = heatmap.shape
            scale_x = heatmap_w / 96
            scale_y = heatmap_h / 96

            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            r_scaled = max(int(r * scale_x), 1)  # radius mag niet nul zijn

            # Use scaled coordinates in the mask
            mask = create_dot_mask(heatmap.shape, x_scaled, y_scaled, r_scaled)

            gmi = np.sum(heatmap * mask) / (np.sum(heatmap) + 1e-8)

            gmi = np.sum(heatmap * mask) / (np.sum(heatmap) + 1e-8)

            results.append({
                "image_name": row["image_name"],
                "x": x,
                "y": y,
                "radius": r,
                "prediction": pred,
                "GMI": gmi
            })
        except Exception as e:
            print(f" Image error {row['image_name']}: {e}")

    return pd.DataFrame(results)


results_df = run_gmi_analysis(
    model_json="cnn_model_augmented data.json",
    weights="cnn_model_augmented data_weights.hdf5",
    csv_path="test_stip_coordinates.csv",
    image_dir=r"D:\School\Project AI for MIA\data\test_jpg",  # Folder with images
    conv_layer_name="conv2d_2"   # Can be changed to another convolutional layer
)

print(results_df.head())
results_df.to_csv("gmi_results_augmented_test_normal.csv", index=False)


