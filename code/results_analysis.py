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


def plot_side_by_side(image_names, base_dir, csv_path, model1, model2, label1="Normal", label2="Augmented"):
    df = pd.read_csv(csv_path)
    df = df[df["image_name"].isin(image_names)].set_index("image_name")

    fig, axs = plt.subplots(len(image_names), 2, figsize=(8, 4 * len(image_names)))

    for i, image_name in enumerate(image_names):
        image_path = os.path.join(base_dir, image_name)
        image = load_image(image_path)
        heatmap1, pred1 = compute_gradcam(model1, image)
        heatmap2, pred2 = compute_gradcam(model2, image)

        overlay1 = overlay_gradcam(image, heatmap1)
        overlay2 = overlay_gradcam(image, heatmap2)

        axs[i, 0].imshow(overlay1)
        axs[i, 0].set_title(f"{label1}\nPred: {pred1:.2f} | GMI: {df.loc[image_name, 'GMI']:.2f}")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(overlay2)
        axs[i, 1].set_title(f"{label2}\nPred: {pred2:.2f} | GMI: N/A")
        axs[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


model_normal = load_model("cnn_model_normal_data.json", "cnn_model_normal_data_weights.hdf5")
model_augmented = load_model("cnn_model_augmented data.json", "cnn_model_augmented data_weights.hdf5")

# Kies 5 afbeeldingen
image_names_to_plot = [
    "00001b2b5609af42ab0ab276dd4cd41c3e7745b5.jpg",
    "0018f45df78db5e21d0b52f4d97a1f6540786aab.jpg",
    "0059d8e5018a03b8af4454b23f5f429476c392e4.jpg",
    "0069beab1504f4dbb63f51682c743ac12c477c4a.jpg",
    "007fcf180a28349702a4a6749e20554e0cb326b2.jpg"
]

plot_side_by_side(
    image_names=image_names_to_plot,
    base_dir=r"D:\School\Project AI for MIA\data\train+val\valid\1_val_modified",
    csv_path="gmi_results_normal.csv",
    model1=model_normal,
    model2=model_augmented
)

results_normal = "gmi_results_normal.csv"
results_augmented = "gmi_results_augmented.csv"
t_stat_analysis(results_normal, results_augmented)