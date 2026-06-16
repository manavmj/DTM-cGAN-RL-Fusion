import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots(train_csv="outputs/eval_results_train.csv", test_csv="outputs/eval_results_test.csv", out_dir="outputs/eval_graphs"):
    os.makedirs(out_dir, exist_ok=True)
    
    # Load and combine data
    dfs = []
    if os.path.exists(train_csv):
        dfs.append(pd.read_csv(train_csv))
    if os.path.exists(test_csv):
        dfs.append(pd.read_csv(test_csv))
        
    if not dfs:
        print("No evaluation CSVs found to plot!")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Set seaborn style
    sns.set_theme(style="whitegrid")
    
    # Metrics to plot distributions for
    metrics = ["PSNR", "SSIM", "LPIPS", "EN", "MI", "AG", "SF", "Latency_ms", "FPS"]
    
    print("Generating distribution plots...")
    for metric in metrics:
        if metric in df.columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=df, x="Split", y=metric, palette="Set2")
            plt.title(f"{metric} Comparison (Train vs Test)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{metric}_comparison.png"))
            plt.close()
            
    # Single value metrics comparison (FID, KID, mAP, Precision, Recall)
    single_metrics = ["FID", "KID", "mAP", "Precision", "Recall"]
    print("Generating bar charts for global metrics...")
    
    summary_df = df.groupby("Split")[single_metrics].mean().reset_index()
    
    for metric in single_metrics:
        if metric in summary_df.columns:
            plt.figure(figsize=(8, 6))
            sns.barplot(data=summary_df, x="Split", y=metric, palette="Set1")
            plt.title(f"{metric} Score (Global/Averaged)")
            
            # Add text labels
            for index, row in summary_df.iterrows():
                plt.text(index, row[metric], round(row[metric], 4), color='black', ha="center", va="bottom")
                
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{metric}_bar.png"))
            plt.close()
            
    # Performance profile (FPS vs GPU Utilization)
    print("Generating performance profiles...")
    if "FPS" in df.columns and "GPU_Memory_MB" in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x="FPS", y="GPU_Memory_MB", hue="Split", style="Split", s=100, palette="deep")
        plt.title("Performance Profile: FPS vs GPU Memory")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "perf_fps_vs_gpu.png"))
        plt.close()
        
    print(f"All plots saved to {out_dir}/")

if __name__ == "__main__":
    generate_plots()
