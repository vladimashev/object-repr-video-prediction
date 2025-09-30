import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
from pathlib import Path

SMOOTHNESS = 100   # try 10, 50, 100 depending on noise
def smooth_series(series, window):
    if window > 1:
        return series.rolling(window, min_periods=1).mean()
    return series

LINEWIDTH = 2
def plot_models(log_files,
               metric_name="loss",
               plt_title=None,
               mode="epoch",
               smoothness=SMOOTHNESS):
    """
    Plots a selected metric for multiple models.

    Arguments:
        log_files: { "model_name": "file_path.csv" } - list of models to plot
        metric_name: str - column name in csv to plot
        plt_title: str - plot name
        mode: "epoch" | "iter" ("epoch" by default)
        smoothness: smoothing window for per-iteration plot
    """
    assert mode in ["epoch", "iter"], 'mode must be either "epoch" or "iter"'

    if plt_title is None:
        plt_title = f"Learning curves for {metric_name}"

    plt.figure(figsize=(10, 6))

    for model_name, file_path in log_files.items():
        df = pd.read_csv(file_path)

        if metric_name not in df.columns:
            raise Exception(f"{metric_name} not found in {file_path}, exitted script.")

        if mode == "epoch":
            # CASE 1: metric is dense (logged every iteration) → average per epoch
            if df.groupby("epoch")[metric_name].count().max() > 1:
                df_epoch = df.groupby("epoch")[metric_name].mean().reset_index()
                plt.plot(df_epoch["epoch"],
                         df_epoch[metric_name],
                         label=f"{model_name}",
                         linewidth=LINEWIDTH)
            else:
                # CASE 2: metric is sparse (logged only every N iterations)
                # → plot at given epoch positions
                plt.plot(df["epoch"],
                         df[metric_name],
                         "o-",
                         label=f"{model_name}",
                         linewidth=LINEWIDTH,
                         alpha=0.8)

        elif mode == "iter":
            # Per-iteration plot (smoothed to reduce noise)
            plt.plot(df["iter"],
                     smooth_series(df[metric_name], smoothness),
                     label=model_name,
                     linewidth=LINEWIDTH)

    if mode == "iter":
        # Replace iteration ticks with epoch numbers
        epoch_iters = df.groupby("epoch")["iter"].min().values
        plt.xticks(epoch_iters, df["epoch"].unique())
        plt.xlabel("Epoch")
    else:
        plt.xlabel("Epoch")

    plt.ylabel(metric_name)
    plt.title(plt_title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    # Force integer ticks on epoch axis
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()

    # Directory to save plots: to the same folder in which script is contained
    cur_dir = Path(__file__).resolve().parent
    plot_dir = os.path.join(cur_dir, "cached_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Save figure
    cur_time = datetime.now().strftime("%H-%M_%d-%m-%Y")
    plt.savefig(f"{plot_dir}/{cur_time}_MODELS_{'-'.join([str(x) for x in log_files.keys()])}.png")

if __name__ == "__main__":
    # EXP_DIR = 'experiments'
    # log_files = {
    #     "weight 0.0": f"{EXP_DIR}/29-09-2025_10-32_Masked_AE_0.0/logs/metrics.csv",
    #     "weight 0.25": f"{EXP_DIR}/29-09-2025_12-25_Masked_AE_25.0/logs/metrics.csv",
    #     "weight 0.5": f"{EXP_DIR}/29-09-2025_13-05_Masked_AE_50.0/logs/metrics.csv",
    #     "weight 0.75": f"{EXP_DIR}/29-09-2025_13-45_Masked_AE_75.0/logs/metrics.csv",
    #     "weight 1.0": f"{EXP_DIR}/29-09-2025_14-24_Masked_AE_100.0/logs/metrics.csv",
    # }

    EXP_DIR = 'experiments_released'
    log_files = {
        "Embed dim 512": f"{EXP_DIR}/28-09-2025_22-18_Patch_AE/logs/metrics.csv",
        "Embed dim 256": f"{EXP_DIR}/29-09-2025_00-15_Patch_AE/logs/metrics.csv",
        "Embed dim 128": f"{EXP_DIR}/29-09-2025_05-36_Patch_AE/logs/metrics.csv",
    }
    # log_files = {
    #     "Embed dim 512": f"{EXP_DIR}/28-09-2025_14-21_Mask_AE/logs/metrics.csv",
    #     "Embed dim 256": f"{EXP_DIR}/29-09-2025_06-32_Masked_AE/logs/metrics.csv",
    #     "Embed dim 128": f"{EXP_DIR}/29-09-2025_07-36_Masked_AE/logs/metrics.csv",
    # }

    plot_models(log_files,
               plt_title="Training losses for Patched AE",
               metric_name="loss")

    print("Done!")
