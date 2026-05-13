"""
run_xception_inference.py — P2 Xception Mini Evaluation
用 timm Xception 直接跑 inference，輸出 scores_visual_xception_v0.csv

Modified to support UCF Detector and other detectors
"""

import os
import csv
import time
import yaml
import random
import torch
import timm
import argparse
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score

from detectors import DETECTOR

METADATA_CSV = "/Users/nia/DeepfakeBench/evaluation/mini_eval_metadata.csv"
OUTPUT_CSV   = "/Users/nia/DeepfakeBench/evaluation/scores_visual_xception_v0.csv"
FRAMES_PER_CLIP = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add argument parser
parser = argparse.ArgumentParser(description='Evaluate deepfake detector')
parser.add_argument('--detector_path', type=str, 
                    default='./training/config/detector/xception.yaml',
                    help='path to detector YAML file')
parser.add_argument('--weights_path', type=str, 
                    default=None,
                    help='path to model weights')
parser.add_argument('--metadata_csv', type=str,
                    default=METADATA_CSV,
                    help='path to metadata CSV')
parser.add_argument('--output_csv', type=str,
                    default=OUTPUT_CSV,
                    help='path to output CSV')
parser.add_argument('--detector_type', type=str,
                    default='xception',
                    help='detector type (xception, ucf, etc)')
args = parser.parse_args()

def load_model():
    """Load model based on detector type and configuration"""
    
    # Load detector config
    with open(args.detector_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    detector_type = config.get('model_name', args.detector_type)
    
    if detector_type == 'xception':
        # Legacy: timm xception with ImageNet weights
        print(f"Loading Xception (timm, ImageNet pretrained)...")
        model = timm.create_model('xception', pretrained=True, num_classes=2)
        model = model.to(DEVICE)
    else:
        # Load custom detector (UCF, etc)
        print(f"Loading {detector_type.upper()} detector from config...")
        try:
            model_class = DETECTOR[detector_type]
            model = model_class(config).to(DEVICE)
            
            # Load weights if provided
            if args.weights_path:
                print(f"Loading weights from {args.weights_path}...")
                ckpt = torch.load(args.weights_path, map_location=DEVICE)
                model.load_state_dict(ckpt, strict=True)
                print('===> Load checkpoint done!')
            else:
                print('Warning: No weights path provided')
                
        except KeyError:
            raise ValueError(f"Detector type '{detector_type}' not found in DETECTOR registry")
    
    model.eval()
    return model, detector_type

def predict_clip(model, frame_dir, detector_type):
    """Predict fake score for a clip"""
    frames_files = sorted(os.listdir(frame_dir))
    indices = [int(i * (len(frames_files) - 1) / (FRAMES_PER_CLIP - 1)) for i in range(FRAMES_PER_CLIP)]
    selected = [frames_files[i] for i in indices]

    scores = []
    t_start = time.perf_counter()
    
    if detector_type == 'xception':
        # Xception (timm) prediction
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        for fname in selected:
            img = Image.open(os.path.join(frame_dir, fname)).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(tensor)  # (1, 2)
                prob = torch.softmax(logits, dim=1)
                scores.append(float(prob[0][1]))
    else:
        # Other detectors (UCF, etc)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        for fname in selected:
            img = Image.open(os.path.join(frame_dir, fname)).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                # Prepare data dict for detector forward pass
                data_dict = {
                    'image': tensor,
                    'label': torch.tensor([0]).to(DEVICE),  # dummy label
                }
                predictions = model(data_dict, inference=True)
                
                # Extract fake score (probability of fake class)
                if 'cls_pred' in predictions:
                    prob = torch.softmax(predictions['cls_pred'], dim=1)
                    scores.append(float(prob[0][1]))
                elif 'prob' in predictions:
                    scores.append(float(predictions['prob'][0][1]))
                else:
                    raise ValueError(f"Cannot find classification output in predictions: {predictions.keys()}")

    elapsed = (time.perf_counter() - t_start) * 1000
    return round(sum(scores) / len(scores), 4), round(elapsed, 2)

def main():
    print("=" * 50)
    print(f"Detector Type: {args.detector_type}")
    print(f"Detector Config: {args.detector_path}")
    print(f"Weights Path: {args.weights_path}")
    print(f"Device: {DEVICE}")
    print("=" * 50)
    
    model, detector_type = load_model()
    print("✓ Model loaded successfully")

    metadata_csv = args.metadata_csv
    output_csv = args.output_csv
    
    with open(metadata_csv) as f:
        samples = list(csv.DictReader(f))
    print(f"共 {len(samples)} 個樣本，開始 inference...")

    rows = []
    for i, sample in enumerate(samples):
        try:
            fake_score, inference_time_ms = predict_clip(model, sample["path"], detector_type)
            status = "ok"
            error = ""
        except Exception as e:
            fake_score, inference_time_ms, status, error = "N/A", 0, "failed", str(e)

        rows.append({
            "sample_id": sample["sample_id"],
            "dataset": sample["dataset"],
            "label": int(sample["label"]),
            "detector_name": f"{detector_type.upper()}_deepfake" if detector_type != 'xception' else "Xception_timm_imagenet",
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
            print(f"  [{i+1}/{len(samples)}] {sample['sample_id']}: fake_score={fake_score}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    y_true = [int(r["label"]) for r in ok_rows]
    y_score = [float(r["fake_score"]) for r in ok_rows]
    auc = roc_auc_score(y_true, y_score)

    print(f"\nAUC: {auc:.4f}")
    print(f"完成：{len(rows)} 個樣本 → {output_csv}")

if __name__ == "__main__":
    main()