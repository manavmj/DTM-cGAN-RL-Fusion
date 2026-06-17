import os
import sys

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import load_config

sns.set_theme(style="whitegrid")

def plot_bar(df, x_col, y_col, title, ylabel, out_path):
    plt.figure(figsize=(10, 6))
    order = df.groupby(x_col)[y_col].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x=x_col, y=y_col, order=order, palette="viridis")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_line(df, x_col, y_col, title, ylabel, out_path):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=x_col, y=y_col, marker="o", color="blue")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(x_col)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_scatter(df, x_col, y_col, hue_col, title, xlabel, ylabel, out_path):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, s=150, palette="deep", marker="X")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def get_external_baselines():
    # Placeholders for external baselines (to be filled manually later)
    baselines = ["DenseFuse", "U2Fusion", "RFN-Nest", "SwinFusion", "CDDFuse", "RL-FusionMamba"]
    data = []
    for b in baselines:
        # Dummy placeholder values for graphs
        data.append({
            "Method": b + " (Placeholder)",
            "SSIM": np.random.uniform(0.6, 0.8),
            "MI": np.random.uniform(2.0, 3.5),
            "EN": np.random.uniform(6.0, 7.5),
            "PSNR": np.random.uniform(15.0, 25.0),
            "Approx_FMI": np.random.uniform(0.8, 1.0),
            "mAP50_8n": np.random.uniform(0.5, 0.8),
            "Precision_8n": np.random.uniform(0.6, 0.85),
            "Recall_8n": np.random.uniform(0.5, 0.8),
            "F1_8n": np.random.uniform(0.55, 0.82),
            "FPS": np.random.uniform(10, 40),
            "FLOPs_G": np.random.uniform(20, 50),
            "Latency_ms": np.random.uniform(25, 100),
            "GPU_Memory_MB": np.random.uniform(1000, 3000)
        })
    return pd.DataFrame(data)

def main():
    os.makedirs("outputs/plots", exist_ok=True)
    
    # 1. Evaluate Metrics (Comparison)
    if os.path.exists("outputs/eval_results_test.csv"):
        eval_df = pd.read_csv("outputs/eval_results_test.csv")
        print("Generating performance comparisons...")
        
        agg_df = eval_df.groupby("Method").mean(numeric_only=True).reset_index()
        
        # Add external baseline placeholders
        external_df = get_external_baselines()
        combined_df = pd.concat([agg_df, external_df], ignore_index=True)
        
        # Fusion Quality
        for metric in ["SSIM", "MI", "EN", "PSNR", "Approx_FMI"]:
            if metric in combined_df.columns:
                plot_bar(combined_df, "Method", metric, f"{metric} Comparison", metric, f"outputs/plots/{metric}_Comparison.png")
            
        # Detection
        for metric in ["mAP50_8n", "Precision_8n", "Recall_8n", "F1_8n"]:
            if metric in combined_df.columns:
                plot_bar(combined_df, "Method", metric, f"{metric} Comparison", metric, f"outputs/plots/{metric}_Comparison.png")
            
        # Latency
        for metric in ["FPS", "FLOPs_G", "Latency_ms", "GPU_Memory_MB"]:
            if metric in combined_df.columns:
                plot_bar(combined_df, "Method", metric, f"{metric} Comparison", metric, f"outputs/plots/{metric}_Comparison.png")
            
        # Trade-off Analysis
        if "Latency_ms" in combined_df.columns and "mAP50_8n" in combined_df.columns:
            plot_scatter(combined_df, "Latency_ms", "mAP50_8n", "Method", "mAP vs Latency Trade-off", "Latency (ms)", "mAP@0.5 (YOLOv8n)", "outputs/plots/mAP_vs_Latency.png")
        if "Latency_ms" in combined_df.columns and "SSIM" in combined_df.columns:
            plot_scatter(combined_df, "Latency_ms", "SSIM", "Method", "SSIM vs Latency Trade-off", "Latency (ms)", "SSIM", "outputs/plots/SSIM_vs_Latency.png")
        if "FLOPs_G" in combined_df.columns and "mAP50_8n" in combined_df.columns:
            plot_scatter(combined_df, "FLOPs_G", "mAP50_8n", "Method", "FLOPs vs Accuracy Trade-off", "FLOPs (G)", "mAP@0.5 (YOLOv8n)", "outputs/plots/FLOPs_vs_Accuracy.png")

        print("Evaluation comparison plots generated.")

    # 2. PPO Training Logs
    if os.path.exists("outputs/ppo_training_logs.csv"):
        train_df = pd.read_csv("outputs/ppo_training_logs.csv")
        
        # RL Performance
        plot_line(train_df, "Epoch", "PPO_Reward", "Reward vs Episodes", "Reward", "outputs/plots/Reward_vs_Episodes.png")
        plot_line(train_df, "Epoch", "PPO_Policy_Loss", "Policy Loss vs Episodes", "Policy Loss", "outputs/plots/Policy_Loss_vs_Episodes.png")
        plot_line(train_df, "Epoch", "PPO_Value_Loss", "Value Loss vs Episodes", "Value Loss", "outputs/plots/Value_Loss_vs_Episodes.png")
        plot_line(train_df, "Epoch", "PPO_Entropy_Loss", "Entropy Loss vs Episodes", "Entropy Loss", "outputs/plots/Entropy_Loss_vs_Episodes.png")
        
        plot_line(train_df, "Epoch", "Latency_Reduction_Pct", "Latency Reduction vs Training", "Latency Reduction (%)", "outputs/plots/Latency_Reduction_vs_Training.png")
        plot_line(train_df, "Epoch", "Accuracy_Improvement_Pct", "Accuracy Improvement vs Training", "Accuracy Improvement (%)", "outputs/plots/Accuracy_Improvement_vs_Training.png")
        
        print("RL performance training plots generated.")

if __name__ == "__main__":
    main()
