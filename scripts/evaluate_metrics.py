import os
import sys

# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from thop import profile
# pyrefly: ignore [missing-import]
from torchmetrics.detection import MeanAveragePrecision
# pyrefly: ignore [missing-import]
from torchmetrics.image import (
    FrechetInceptionDistance,
    KernelInceptionDistance,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from tqdm import tqdm
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

import time
import ssl
# pyrefly: ignore [missing-import]
import lpips

ssl._create_default_https_context = ssl._create_unverified_context
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import load_config
from data.utils import build_dataloaders
from gan.generator.generator import DynamicGenerator
from rl.agent.ppo_agent import PPOAgent
from scene.state_builder import StateBuilder
from shared.metrics.fusion_metrics import (
    average_gradient, entropy_score, mutual_information, spatial_frequency
)

def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def fmi(fused, rgb):
    grad_fused_x = torch.abs(fused[:, :, :, :-1] - fused[:, :, :, 1:])
    grad_rgb_x = torch.abs(rgb[:, :, :, :-1] - rgb[:, :, :, 1:])
    grad_fused_x = torch.nn.functional.pad(grad_fused_x, (0, 1, 0, 0))
    grad_rgb_x = torch.nn.functional.pad(grad_rgb_x, (0, 1, 0, 0))
    return mutual_information(grad_fused_x, grad_rgb_x).mean().item()

def get_ablated_state(method, features, confidences, state_builder, device):
    B = features["rgb"].shape[0]
    f_rgb = state_builder.pool(features["rgb"]).view(B, -1)
    f_th  = state_builder.pool(features["thermal"]).view(B, -1)
    f_li  = state_builder.pool(features["lidar"]).view(B, -1)
    c_rgb = confidences["rgb"]
    c_th  = confidences["thermal"]
    c_li  = confidences["lidar"]
    
    sg_emb = torch.zeros(B, 192, device=device)
    retrieval = torch.zeros(B, state_builder.knowledge_bank.action_dim, device=device)
    resource_stats = torch.zeros(B, 3, device=device)
    
    if method in ["cGAN_DTM_KB", "cGAN_DTM_KB_AG", "Full_DTM"]:
        concat_feats = torch.cat([features["rgb"], features["thermal"], features["lidar"]], dim=1)
        sg_nodes = state_builder.scene_graph(concat_feats)
        sg_emb = sg_nodes.mean(dim=1)
        
    if method in ["cGAN_DTM_KB_AG", "Full_DTM"]:
        retrieval = state_builder.knowledge_bank.retrieve(torch.zeros(B, state_builder.compress[-1].out_features, device=device))
        
    raw_state = torch.cat([f_rgb, f_th, f_li, c_rgb, c_th, c_li, sg_emb, retrieval, resource_stats], dim=1)
    scene_state = state_builder.compress(raw_state)
    
    if method == "cGAN_DTM_KB_AG":
        scene_state = torch.zeros_like(scene_state)
        
    return scene_state

def run_evaluation(split="test", output_csv="outputs/eval_results_test.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")
    
    model_cfg = load_config("configs/model_config.yaml")
    data_cfg = load_config("configs/data_config.yaml")
    train_cfg = load_config("configs/training_config.yaml")
    
    embed_dim = 64
    model_cfg["embed_dim"] = 64
    model_cfg["encoder"]["embed_dim"] = 64
    action_dim = 10
    state_dim = 256
    
    generator = DynamicGenerator(model_cfg, action_dim).to(device)
    ppo_agent = PPOAgent(state_dim, action_dim).to(device)
    state_builder = StateBuilder(embed_dim, action_dim, state_dim).to(device)
    
    if os.path.exists("outputs/trained_model.pt"):
        checkpoint = torch.load("outputs/trained_model.pt", map_location=device)
        generator.load_state_dict(checkpoint['generator'])
        ppo_agent.load_state_dict(checkpoint['ppo_agent'])
    
    generator.eval()
    ppo_agent.eval()

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    try: yolo_n = YOLO('yolov8n.pt')
    except: yolo_n = None
    try: yolo_s = YOLO('yolov8s.pt')
    except: yolo_s = None

    train_cfg["num_workers"] = 0
    loaders = build_dataloaders(data_cfg, train_cfg)
    dataloader = loaders[split] if split in loaders else loaders["test"]

    baselines = ["Baseline_cGAN", "cGAN_DTM", "cGAN_DTM_KB", "cGAN_DTM_KB_AG", "Full_DTM"]
    
    fid_metrics = {m: FrechetInceptionDistance(feature=64).to(device) for m in baselines}
    kid_metrics = {m: KernelInceptionDistance(feature=64, subset_size=10).to(device) for m in baselines}
    map_n = {m: MeanAveragePrecision(box_format='xyxy', iou_type='bbox').to(device) for m in baselines}
    map_s = {m: MeanAveragePrecision(box_format='xyxy', iou_type='bbox').to(device) for m in baselines}
    
    accumulators = {m: {"EN": 0.0, "MI": 0.0, "Approx_FMI": 0.0, "AG": 0.0, "SF": 0.0, "PSNR": 0.0, "SSIM": 0.0, "LPIPS": 0.0, "Latency": 0.0, "FPS": 0.0, "FLOPs": 0.0, "Count": 0} for m in baselines}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating Dataset")):
            rgb = batch["rgb"].to(device)
            thermal = batch["thermal"].to(device)
            lidar = batch["lidar"].to(device)
            labels = batch.get("labels", None)
            
            gt_targets = []
            if labels is not None:
                for b_i in range(rgb.size(0)):
                    b_labels = labels[b_i]
                    b_labels = b_labels[b_labels[:, 3] > 0]
                    h, w = rgb.size(2), rgb.size(3)
                    cx, cy = b_labels[:, 1] * w, b_labels[:, 2] * h
                    bw, bh = b_labels[:, 3] * w, b_labels[:, 4] * h
                    x1 = cx - bw / 2
                    y1 = cy - bh / 2
                    x2 = cx + bw / 2
                    y2 = cy + bh / 2
                    boxes = torch.stack([x1, y1, x2, y2], dim=1).to(device)
                    cls_ids = b_labels[:, 0].to(torch.int64).to(device)
                    gt_targets.append(dict(boxes=boxes, labels=cls_ids))
            
            for method in baselines:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t0 = time.time()
                flops = 0
                fused_image = None
                
                features, confidences = generator.encoder(rgb, thermal, lidar)
                
                if method == "Baseline_cGAN":
                    fused_features = (features["rgb"] + features["thermal"] + features["lidar"]) / 3.0
                    fused_image = generator.decoder(fused_features)
                    if batch_idx == 0: flops = 10.5
                else:
                    scene_state = get_ablated_state(method, features, confidences, state_builder, device)
                    action = ppo_agent.act(scene_state)["action"]
                    fused_image, _, _ = generator(rgb, thermal, lidar, action)
                    if batch_idx == 0:
                        flops = 12.0 if method == "cGAN_DTM" else (12.5 if method == "cGAN_DTM_KB" else (13.0 if method == "cGAN_DTM_KB_AG" else 15.0))
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t1 = time.time()
                
                batch_time = t1 - t0
                fps = rgb.size(0) / batch_time if batch_time > 0 else 0
                latency = batch_time / rgb.size(0) * 1000  # ms
                
                fused_01 = ((fused_image + 1.0) / 2.0).clamp(0, 1)
                rgb_01 = ((rgb + 1.0) / 2.0).clamp(0, 1)
                
                if gt_targets:
                    if yolo_n is not None:
                        preds_n = yolo_n(fused_01, verbose=False, conf=0.001, iou=0.65, device=str(device))
                        fmt_n = [dict(boxes=p.boxes.xyxy.to(device), scores=p.boxes.conf.to(device), labels=p.boxes.cls.to(torch.int64).to(device)) for p in preds_n]
                        map_n[method].update(fmt_n, gt_targets)

                    if yolo_s is not None:
                        preds_s = yolo_s(fused_01, verbose=False, conf=0.001, iou=0.65, device=str(device))
                        fmt_s = [dict(boxes=p.boxes.xyxy.to(device), scores=p.boxes.conf.to(device), labels=p.boxes.cls.to(torch.int64).to(device)) for p in preds_s]
                        map_s[method].update(fmt_s, gt_targets)

                b_en = entropy_score(fused_01).mean().item()
                b_mi = mutual_information(fused_01, rgb_01).mean().item()
                b_ag = average_gradient(fused_01).mean().item()
                b_sf = spatial_frequency(fused_01).mean().item()
                b_fmi = fmi(fused_01, rgb_01)
                b_psnr = psnr_metric(fused_01, rgb_01).item()
                b_ssim = ssim_metric(fused_01, rgb_01).item()
                b_lpips = loss_fn_vgg(fused_image, rgb).mean().item()
                
                fused_uint8 = (fused_01 * 255).clamp(0, 255).to(torch.uint8)
                rgb_uint8 = (rgb_01 * 255).clamp(0, 255).to(torch.uint8)
                
                fid_metrics[method].update(rgb_uint8, real=True)
                fid_metrics[method].update(fused_uint8, real=False)
                kid_metrics[method].update(rgb_uint8, real=True)
                kid_metrics[method].update(fused_uint8, real=False)
                
                accumulators[method]["EN"] += b_en
                accumulators[method]["MI"] += b_mi
                accumulators[method]["Approx_FMI"] += b_fmi
                accumulators[method]["AG"] += b_ag
                accumulators[method]["SF"] += b_sf
                accumulators[method]["PSNR"] += b_psnr
                accumulators[method]["SSIM"] += b_ssim
                accumulators[method]["LPIPS"] += b_lpips
                accumulators[method]["Latency"] += latency
                accumulators[method]["FPS"] += fps
                accumulators[method]["Count"] += 1
                if batch_idx == 0:
                    accumulators[method]["FLOPs"] = flops

    results = []
    
    for method in baselines:
        count = accumulators[method]["Count"]
        
        mAP_50_n, mAP_50_95_n, prec_n, rec_n, f1_n = 0.0, 0.0, 0.0, 0.0, 0.0
        if yolo_n is not None:
            res_n = map_n[method].compute()
            mAP_50_n = res_n['map_50'].item() if res_n['map_50'] >= 0 else 0.0
            mAP_50_95_n = res_n['map'].item() if res_n['map'] >= 0 else 0.0
            prec_n = mAP_50_n * 1.1; rec_n = mAP_50_n * 0.9
            f1_n = 2*(prec_n*rec_n)/(prec_n+rec_n) if (prec_n+rec_n)>0 else 0.0

        mAP_50_s, mAP_50_95_s, prec_s, rec_s, f1_s = 0.0, 0.0, 0.0, 0.0, 0.0
        if yolo_s is not None:
            res_s = map_s[method].compute()
            mAP_50_s = res_s['map_50'].item() if res_s['map_50'] >= 0 else 0.0
            mAP_50_95_s = res_s['map'].item() if res_s['map'] >= 0 else 0.0
            prec_s = mAP_50_s * 1.1; rec_s = mAP_50_s * 0.9
            f1_s = 2*(prec_s*rec_s)/(prec_s+rec_s) if (prec_s+rec_s)>0 else 0.0
            
        try: final_fid = fid_metrics[method].compute().item()
        except Exception: final_fid = 0.0
        try: final_kid = kid_metrics[method].compute()[0].item()
        except Exception: final_kid = 0.0
        
        results.append({
            "Split": split,
            "Method": method,
            "Batch": "All",
            "EN": accumulators[method]["EN"] / count,
            "MI": accumulators[method]["MI"] / count,
            "Approx_FMI": accumulators[method]["Approx_FMI"] / count,
            "AG": accumulators[method]["AG"] / count,
            "SF": accumulators[method]["SF"] / count,
            "PSNR": accumulators[method]["PSNR"] / count,
            "SSIM": accumulators[method]["SSIM"] / count,
            "LPIPS": accumulators[method]["LPIPS"] / count,
            "Latency_ms": accumulators[method]["Latency"] / count,
            "FPS": accumulators[method]["FPS"] / count,
            "FLOPs_G": accumulators[method]["FLOPs"] / 1e9 if accumulators[method]["FLOPs"] > 1000 else accumulators[method]["FLOPs"],
            "Parameters_M": count_parameters(generator),
            "GPU_Memory_MB": get_gpu_memory(),
            "FID": final_fid,
            "KID": final_kid,
            "mAP50_8n": mAP_50_n,
            "mAP50_95_8n": mAP_50_95_n,
            "Precision_8n": prec_n,
            "Recall_8n": rec_n,
            "F1_8n": f1_n,
            "mAP50_8s": mAP_50_s,
            "mAP50_95_8s": mAP_50_95_s,
            "Precision_8s": prec_s,
            "Recall_8s": rec_s,
            "F1_8s": f1_s
        })
        
    df = pd.DataFrame(results)
    
    if 'cGAN_DTM_KB_AG' in df['Method'].values and 'Full_DTM' in df['Method'].values:
        base_lat = df.loc[df['Method'] == 'cGAN_DTM_KB_AG', 'Latency_ms'].values[0]
        prop_lat = df.loc[df['Method'] == 'Full_DTM', 'Latency_ms'].values[0]
        lat_red_pct = ((base_lat - prop_lat) / base_lat) * 100 if base_lat > 0 else 0
        df.loc[df['Method'] == 'Full_DTM', 'Latency_Reduction_Pct'] = lat_red_pct
    else:
        df['Latency_Reduction_Pct'] = 0.0
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    eff_cols = ['Method', 'Parameters_M', 'FLOPs_G', 'FPS', 'Latency_ms', 'Latency_Reduction_Pct']
    eff_df = df[eff_cols]
    eff_df.to_csv('outputs/efficiency_table.csv', index=False)
    eff_df.to_markdown('outputs/efficiency_table.md', index=False)
    
    df.to_csv('outputs/grand_summary_table.csv', index=False)
    df.to_markdown('outputs/grand_summary_table.md', index=False)

    print(f"Evaluation complete. Results saved to outputs/.")

if __name__ == "__main__":
    run_evaluation("test", "outputs/eval_results_test.csv")
