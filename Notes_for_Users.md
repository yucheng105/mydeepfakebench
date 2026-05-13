# Newest Release of DeepfakeBench

# 啟動
記得要先開 Docker Desktop
## Docker Build - 做出 image
```bash
docker build -t deepfakebench .
```

## Docker run - image -> container
```bash
docker run --gpus all -itd -v "$(Get-Location):/app" --shm-size 64G --name deepfakebench-container deepfakebench
```

## Docker start - 啟動 container
```bash
docker start deepfakebench-container
```

## Docker exec - 進入 container 執行指令
```bash
docker exec -it deepfakebench-container /bin/bash
```

# Testing
## Terminal 輸入
```bash
python training/test.py --detector_path ./training/config/detector/ucf.yaml --test_dataset "Celeb-DF-v2" --weights_path ./training/weights/ucf_best.pth
```
可以換的地方：
* Detector
* Weights
## 資料集指定
至 training/config/test_config.yaml 做修改。
目前是用絕對路徑。

# Eval: 輸出.csv
```bash
python training/eval.py \
  --detector_path ./training/config/detector/ucf.yaml \
  --weights_path ./training/weights/ucf_best.pth \
  --detector_type ucf \
  --metadata_csv ./evaluation/mini_eval_metadata.csv \
  --output_csv ./evaluation/scores_ucf_v0.csv
```

# Training
```bash
python training/train.py --detector_path ./training/config/detector/facexray.yaml --train_dataset "FaceForensics++" --test_dataset "FaceForensics++"
```

## Notes to Myself
* **可以輸出分數**：終於可以成功輸出分數了。
* **分數看來可疑**：分數太低了，要再確認一下權重要從哪裡取得。論文？Official Repo？
* **研究一下 config**：要研究如何抽換測試資料集。
* **檢查 GPU 使用率**：
```bash
nvidia-smi -l 1
```