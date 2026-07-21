import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_losses(work_dir: str,rank: int = None):

    print(f'work_dir: {work_dir}')
    if rank is not None:
        csv_path = os.path.join(work_dir, f"train_{rank}.csv")
    else:
        csv_path = os.path.join(work_dir, "train.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    index_fist_over_1000 = next((i for i, x in enumerate(df["frame"]) if x > 1000), None)

    if index_fist_over_1000 is not None:

        plt.figure(figsize=(8, 5))
        plt.plot(df["frame"].iloc[index_fist_over_1000:], df["actor_loss"].iloc[index_fist_over_1000:], label="actor_loss")
        if "img_loss" in df.columns:
            plt.plot(df["frame"].iloc[index_fist_over_1000:], df["img_loss"].iloc[index_fist_over_1000:], label="img_loss")

        plt.xlabel("frame")
        plt.ylabel("loss")
        if "img_loss" in df.columns:
            plt.title("Actor Loss & Img Loss vs Frame")
        else:
            plt.title("Actor Loss vs Frame")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        if rank is not None:
            out_path = os.path.join(work_dir, f"loss_fig_{rank}.png")
        else:
            out_path = os.path.join(work_dir, "loss_fig.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"Figure saved to: {out_path}")

def plot_eval_losses(work_dir):
    csv_path = os.path.join(work_dir, f"eval_loss.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path)

    index_fist_over_1000 = next((i for i, x in enumerate(df["global_step"]) if x > 1000), None)

    if index_fist_over_1000 is not None:

        x = df.iloc[:, 0]
        y_cols = df.columns[1:]

        plt.figure(figsize=(10, 6))
        for col in y_cols:
            plt.plot(x.iloc[index_fist_over_1000:], df[col].iloc[index_fist_over_1000:], label=col, linewidth=1.5)
        plt.xlabel("Frames")
        plt.ylabel("Value")
        plt.title(f"{os.path.basename(csv_path)} - Each Task")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        total = df[y_cols].sum(axis=1)
        plt.figure(figsize=(10, 6))
        plt.plot(x.iloc[index_fist_over_1000:], total.iloc[index_fist_over_1000:], color='tab:red', linewidth=2, label='Sum of all tasks')
        plt.xlabel("Frames")
        plt.ylabel("Total Value")
        plt.title(f"{os.path.basename(csv_path)} - Sum of All Tasks")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        plt.savefig(os.path.join(work_dir, "eval_loss_fig.png"), dpi=300)

def plot_eval_losses_series2(work_dir):
    csv_path = os.path.join(work_dir, f"eval_loss.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path)

    index_fist_over_1000 = next((i for i, x in enumerate(df["global_step"]) if x > 1000), None)

    if index_fist_over_1000 is not None:

        x = df.iloc[:, 0]
        y_cols = df.columns[1:]

        plt.figure(figsize=(10, 6))
        for col in y_cols:
            plt.plot(x.iloc[index_fist_over_1000:], df[col].iloc[index_fist_over_1000:], label=col, linewidth=1.5)
        plt.xlabel("Frames")
        plt.ylabel("Value")
        plt.title(f"{os.path.basename(csv_path)} - Each Task")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir, "eval_loss_fig_each_task.png"), dpi=300)

        total = df[y_cols].sum(axis=1)
        plt.figure(figsize=(10, 6))
        plt.plot(x.iloc[index_fist_over_1000:], total.iloc[index_fist_over_1000:], color='tab:red', linewidth=2, label='Sum of all tasks')
        plt.xlabel("Frames")
        plt.ylabel("Total Value")
        plt.title(f"{os.path.basename(csv_path)} - Sum of All Tasks")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        plt.savefig(os.path.join(work_dir, "eval_loss_fig.png"), dpi=300)

def plot_losses_autoencoder(work_dir: str,rank: int = None):

    print(f'work_dir: {work_dir}')
    if rank is not None:
        csv_path = os.path.join(work_dir, f"train_{rank}.csv")
    else:
        csv_path = os.path.join(work_dir, "train.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    num_rows = len(df) - 1

    start_row = None
    for i in range(num_rows):
        if df.iloc[i]["frame"] > 2000:
            start_row = i
            break
    if start_row is not None:
        df = df.iloc[start_row:]

        plt.figure(figsize=(8, 5))
        plt.plot(df["frame"], df["lang_loss"], label="lang_loss")
        if "img_loss" in df.columns:
            plt.plot(df["frame"], df["img_loss"], label="img_loss")

        plt.xlabel("frame")
        plt.ylabel("loss")
        if "img_loss" in df.columns:
            plt.title("Lang Loss & Img Loss vs Frame")
        else:
            plt.title("Lang Loss vs Frame")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        if rank is not None:
            out_path = os.path.join(work_dir, f"loss_fig_{rank}.png")
        else:
            out_path = os.path.join(work_dir, "loss_fig.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"Figure saved to: {out_path}")
