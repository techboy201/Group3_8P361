import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import os
from Grad_cam import load_model, load_image, compute_gradcam, overlay_gradcam


def t_stat_analysis(results_normal, results_augmented):
    # Load both CSVs
    df_normal = pd.read_csv(results_normal)
    df_augmented = pd.read_csv(results_augmented)

    # Add model label
    df_normal["Model"] = "Normal"
    df_augmented["Model"] = "Augmented"

    # Filter on predictions != 1.0
    df_normal_filtered = df_normal[df_normal["prediction"] != 1.0]
    df_augmented_filtered = df_augmented[df_augmented["prediction"] != 1.0]

    # Combine into one DataFrame
    df_combined = pd.concat([df_normal_filtered, df_augmented_filtered], ignore_index=True)

    # Plot boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Model", y="GMI", data=df_combined)
    plt.title("GMI Comparison between Models")
    plt.ylabel("GMI")
    plt.show()

    # Compute and print stats
    mean_normal = df_normal_filtered["GMI"].mean()
    mean_augmented = df_augmented_filtered["GMI"].mean()
    std_normal = df_normal_filtered["GMI"].std()
    std_augmented = df_augmented_filtered["GMI"].std()

    t_stat, p_value = ttest_ind(df_normal_filtered["GMI"], df_augmented_filtered["GMI"], equal_var=False)

    print("=== GMI Comparison (Filtered) ===")
    print(f"Normal Model   -> Mean: {mean_normal:.4f} | Std: {std_normal:.4f}")
    print(f"Augmented Model-> Mean: {mean_augmented:.4f} | Std: {std_augmented:.4f}")
    print(f"T-statistic: {t_stat:.4f} | P-value: {p_value:.4e}")


def plot_side_by_side(image_names, base_dir, csv_normal, csv_augmented, model1, model2, label1="Normal", label2="Augmented"):
    df_normal = pd.read_csv(csv_normal).set_index("image_name")
    df_augmented = pd.read_csv(csv_augmented).set_index("image_name")

    fig, axs = plt.subplots(len(image_names), 2, figsize=(10, 4 * len(image_names)))

    for i, image_name in enumerate(image_names):
        image_path = os.path.join(base_dir, image_name)
        image = load_image(image_path)

        heatmap1, pred1 = compute_gradcam(model1, image)
        heatmap2, pred2 = compute_gradcam(model2, image)

        overlay1 = overlay_gradcam(image, heatmap1)
        overlay2 = overlay_gradcam(image, heatmap2)

        gmi1 = df_normal.loc[image_name, 'GMI'] if image_name in df_normal.index else "N/A"
        gmi2 = df_augmented.loc[image_name, 'GMI'] if image_name in df_augmented.index else "N/A"

        axs[i, 0].imshow(overlay1)
        axs[i, 0].set_title(f"{label1}\nPred: {pred1:.2f} | GMI: {gmi1:.2f}" if gmi1 != "N/A" else f"{label1}\nPred: {pred1:.2f} | GMI: N/A")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(overlay2)
        axs[i, 1].set_title(f"{label2}\nPred: {pred2:.2f} | GMI: {gmi2:.2f}" if gmi2 != "N/A" else f"{label2}\nPred: {pred2:.2f} | GMI: N/A")
        axs[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


model_normal = load_model("cnn_model_normal_data.json", "cnn_model_normadata_weights.hdf5")
model_augmented = load_model("cnn_model_augmented data.json", "cnn_model_augmented data_weights.hdf5")

# Choose 5 images
image_names_to_plot = [
    "0000ec92553fda4ce39889f9226ace43cae3364e.jpg",
    "00024a6dee61f12f7856b0fc6be20bc7a48ba3d2.jpg",
    "000253dfaa0be9d0d100283b22284ab2f6b643f6.jpg",
    "000270442cc15af719583a8172c87cd2bd9c7746.jpg",
    "000360e0d8358db520b5c7564ac70c5706a0beb0.jpg"
]

plot_side_by_side(
    image_names=image_names_to_plot,
    base_dir=r"D:\School\Project AI for MIA\data\test_jpg",
    csv_normal="gmi_results_normal_test.csv",
    csv_augmented="gmi_results_augmented_test_normal.csv",
    model1=model_normal,
    model2=model_augmented
)

results_normal = "gmi_results_normal_test.csv"
results_augmented = "gmi_results_augmented_test_normal.csv"
t_stat_analysis(results_normal, results_augmented)
