import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

LINEWIDTH = 2

def plot_metrics(csv_file,
                 metrics,
                 plt_title="Metrics per Epoch"):
    """
    Plots multiple metrics for a single model.

    Arguments:
        csv_file: str - path to the CSV log file
        metrics: list[str] - list of column names in the CSV to plot
        plt_title: str - title of the plot
    """
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(10, 6))

    for metric in metrics:
        if metric not in df.columns:
            raise Exception(f"Warning: metric '{metric}' not found in CSV, exitted.")

        # Sparse metrics: plot values where they exist
        df_metric = df.dropna(subset=[metric])
        plt.plot(df_metric["epoch"],
                 df_metric[metric],
                 "o-",
                 linewidth=LINEWIDTH,
                 alpha=0.8,
                 label=metric)

    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.title(plt_title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Directory to save plots
    cur_dir = Path(__file__).resolve().parent
    plot_dir = os.path.join(cur_dir, "cached_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Save figure
    cur_time = datetime.now().strftime("%H-%M_%d-%m-%Y")
    metrics_str = "-".join(metrics)
    out_file = f"{plot_dir}/{cur_time}_METRICS_{metrics_str}.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")


if __name__ == "__main__":
    csv_file = "experiments/29-09-2025_14-24_Masked_AE_100.0/logs/metrics.csv"

    metrics_to_plot = ["MAE", "MSE", "Mask_loss"]

    plot_metrics(csv_file,
                 metrics=metrics_to_plot,
                 plt_title="Metrics per Epoch")

    print("Done!")
