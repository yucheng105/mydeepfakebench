"""
run_xception_inference.py — Xception Baseline Evaluation
用 Xception 模型 (training/weights/xception_best.pth) 跑 inference，輸出分數 CSV

Usage:
    python run_xception_inference.py                      # Use mini_eval_metadata.csv (default)
    python run_xception_inference.py celebdfv2            # Use celebdfv2_metadata.csv
    python run_xception_inference.py dfdc                 # Use dfdc_metadata.csv
"""

import os
import sys
import csv
import time
import torch
import yaml
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms
import numpy as np

# 添加 training 目錄到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'training'))

from detectors import DETECTOR  # type: ignore[import-not-found]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LEGACY_DATA_ROOT = '/Users/nia/DeepfakeBench'
EVAL_ROOT = os.path.join(REPO_ROOT, 'evaluation')
METADATA_DIR = os.path.join(EVAL_ROOT, 'metadata')
RESULTS_DIR = os.path.join(EVAL_ROOT, 'results')

# Support different datasets via command-line argument
DATASET_MODE = sys.argv[1] if len(sys.argv) > 1 else 'mini'

# mapping dataset mode -> (metadata filename, output csv name)
METADATA_MAP = {
    'mini': ('mini_eval_metadata.csv', 'scores_visual_xception_v0.csv'),
    'celebdfv2': ('celebdfv2_metadata.csv', 'scores_visual_xception_celebdfv2.csv'),
    'dfdc': ('dfdc_metadata.csv', 'scores_visual_xception_dfdc.csv'),
}

if DATASET_MODE in METADATA_MAP:
    meta_name, out_name = METADATA_MAP[DATASET_MODE]
    METADATA_CSV = os.path.join(METADATA_DIR, meta_name)
    OUTPUT_CSV = os.path.join(RESULTS_DIR, out_name)
else:
    print(f"Unknown dataset mode: {DATASET_MODE} — falling back to 'mini'.")
    METADATA_CSV = os.path.join(METADATA_DIR, 'mini_eval_metadata.csv')
    OUTPUT_CSV = os.path.join(RESULTS_DIR, 'scores_visual_xception_v0.csv')

WEIGHTS_PATH = os.path.join(REPO_ROOT, 'training', 'weights', 'xception_best.pth')
CONFIG_PATH = os.path.join(REPO_ROOT, 'training', 'config', 'detector', 'xception.yaml')
FRAMES_PER_CLIP = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def resolve_frame_dir(frame_path):
    """Map legacy metadata paths to the current workspace layout."""
    normalized_frame_path = frame_path.replace('\\', os.sep).replace('/', os.sep)

    if os.path.exists(normalized_frame_path):
        return normalized_frame_path

    if normalized_frame_path.startswith(LEGACY_DATA_ROOT):
        relative_path = normalized_frame_path[len(LEGACY_DATA_ROOT):].lstrip('/\\')
        candidate = os.path.normpath(os.path.join(REPO_ROOT, relative_path))
        if os.path.exists(candidate):
            return candidate
        return candidate

    candidate = os.path.normpath(os.path.join(REPO_ROOT, normalized_frame_path))
    return candidate if os.path.exists(candidate) else frame_path

def load_config(config_path):
    """載入 Xception 配置檔"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 解析配置中的相對路徑為絕對路徑
    if 'pretrained' in config and config['pretrained']:
        pretrained_path = config['pretrained']
        # 如果是相對路徑，轉換為基於 REPO_ROOT 的絕對路徑
        if not os.path.isabs(pretrained_path):
            config['pretrained'] = os.path.normpath(
                os.path.join(REPO_ROOT, pretrained_path)
            )
            print(f"轉換預訓練路徑: {pretrained_path} → {config['pretrained']}")
    
    return config


def load_model(weights_path, config):
    """
    載入 Xception 模型並讀入預訓練權重
    """
    print(f"初始化 Xception 模型...")
    
    # 從配置建立 Xception 檢測器
    detector_class = DETECTOR['xception']
    model = detector_class(config)
    
    # 載入預訓練權重
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Xception 權重檔不存在: {weights_path}")

    print(f"載入 Xception 權重: {weights_path}")
    state_dict = torch.load(weights_path, map_location=DEVICE)
    
    # 如果權重格式是 DataParallel 的，需要移除 'module.' 前綴
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # 移除 'module.' 前綴
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    print("✓ Xception 權重載入成功")
    
    model = model.to(DEVICE)
    model.eval()
    return model

def predict_clip(model, frame_dir):
    """
    對視頻片段進行推論
    
    Args:
        model: Xception 模型
        frame_dir: 包含視頻幀的目錄
    
    Returns:
        fake_score: 0~1 的 fake 機率
        inference_time_ms: 推論耗時（毫秒）
    """
    frames_files = sorted(os.listdir(frame_dir))
    if len(frames_files) == 0:
        raise ValueError(f"目錄 {frame_dir} 中沒有幀")
    
    # 均勻抽樣 FRAMES_PER_CLIP 幀
    indices = [int(i * (len(frames_files) - 1) / (FRAMES_PER_CLIP - 1)) 
               for i in range(FRAMES_PER_CLIP)]
    selected = [frames_files[i] for i in indices]
    
    scores = []
    t_start = time.perf_counter()
    
    with torch.no_grad():
        for fname in selected:
            try:
                img = Image.open(os.path.join(frame_dir, fname)).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(DEVICE)
                
                # 構造 data_dict（模擬 DeepfakeBench 的格式）
                data_dict = {
                    'image': tensor,
                    'label': torch.zeros(1, device=DEVICE, dtype=torch.long),
                }
                
                # 執行前向傳播（一定要走 inference 分支，避免訓練時的 batch 索引邏輯）
                pred_dict = model(data_dict, inference=True)
                
                # inference mode 只回傳 cls；從 logits 轉成 fake 機率
                if 'cls' in pred_dict:
                    prob = torch.softmax(pred_dict['cls'], dim=1)[:, 1]
                elif 'prob' in pred_dict:
                    prob = pred_dict['prob']
                else:
                    raise KeyError("模型輸出缺少 'cls' 或 'prob' 鍵")
                
                scores.append(float(prob.cpu().item()))
            except Exception as e:
                print(f"  [警告] 處理幀 {fname} 時出錯: {e}")
                continue
    
    if not scores:
        raise ValueError(f"無法從任何幀中提取分數")
    
    elapsed = (time.perf_counter() - t_start) * 1000
    fake_score = round(sum(scores) / len(scores), 4)
    
    return fake_score, round(elapsed, 2)

def main():
    print("=" * 60)
    print("Xception 推論評估")
    print("=" * 60)
    
    # 檢查必要檔案是否存在
    if not os.path.exists(METADATA_CSV):
        print(f"✗ 錯誤: 元數據檔不存在 {METADATA_CSV}")
        return
    
    if not os.path.exists(CONFIG_PATH):
        print(f"✗ 錯誤: 配置檔不存在 {CONFIG_PATH}")
        return
    
    # 載入配置
    print(f"載入配置: {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)
    
    # 載入模型
    model = load_model(WEIGHTS_PATH, config)
    print("✓ 模型載入成功\n")

    with open(METADATA_CSV) as f:
        samples = list(csv.DictReader(f))
    if not samples:
        print(f"✗ 錯誤: 元數據檔是空的，沒有任何樣本: {METADATA_CSV}")
        return
    print(f"共 {len(samples)} 個樣本，開始 inference...\n")

    rows = []
    failed_count = 0
    
    for i, sample in enumerate(samples):
        try:
            frame_dir = resolve_frame_dir(sample["path"])
            if not os.path.exists(frame_dir):
                raise FileNotFoundError(f"幀目錄不存在: {frame_dir}")
            
            fake_score, inference_time_ms = predict_clip(model, frame_dir)
            status = "ok"
            error = ""
        except Exception as e:
            fake_score = "N/A"
            inference_time_ms = 0
            status = "failed"
            error = str(e)
            failed_count += 1

        rows.append({
            "sample_id": sample["sample_id"],
            "dataset": sample.get("dataset", "unknown"),
            "label": int(sample["label"]),
            "detector_name": "Xception",
            "modality": "visual",
            "fake_score": fake_score,
            "score_type": "probability",
            "inference_time_ms": inference_time_ms,
            "window_start_sec": "N/A",
            "window_end_sec": "N/A",
            "status": status,
            "error_message": error,
        })
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] {sample['sample_id']}: "
                  f"fake_score={fake_score}, status={status}")
    
    # 匯出 CSV
    if not rows:
        print("✗ 錯誤: 沒有任何輸出列可寫入，請檢查 metadata 與資料路徑")
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    # 計算 AUC
    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        y_true = [int(r["label"]) for r in ok_rows]
        y_score = [float(r["fake_score"]) for r in ok_rows]
        auc = roc_auc_score(y_true, y_score)
        print(f"\nAUC: {auc:.4f}")
    else:
        print(f"\n⚠ 警告: 沒有成功的推論結果，無法計算 AUC")
    
    print(f"\n完成: {len(rows)} 個樣本 → {OUTPUT_CSV}")
    print(f"成功: {len(ok_rows)}, 失敗: {failed_count}")
    print("=" * 60)

if __name__ == "__main__":
    main()
