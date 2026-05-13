# Celeb-DF-v2 Evaluation Guide

## 目標
使用 `run_ucf_inference.py` 對 Celeb-DF-v2 數據集進行 UCF 深度偽造檢測器評估

## 步驟 1：生成 Celeb-DF-v2 元數據 CSV

首先，列舉 Celeb-DF-v2 目錄結構並生成評估所需的元數據文件：

```bash
cd evaluation/run
python ../metadata/generate_celebdfv2_metadata.py
```

**輸出：**
- `evaluation/metadata/celebdfv2_metadata.csv` — 包含所有 Celeb-DF-v2 樣本的元數據
  - Celeb-real（真實視頻）樣本標籤為 0
  - Celeb-synthesis（合成視頻）樣本標籤為 1
  - 自動計算的幀數和時長信息

## 步驟 2：運行 UCF 推理

### 2.1 使用預生成的 Celeb-DF-v2 元數據運行評估

從 `evaluation` 目錄運行：

```bash
cd evaluation/run
python run_ucf_inference.py celebdfv2
```

**說明：**
- 參數 `celebdfv2` 告訴腳本使用 `celebdfv2_metadata.csv`
- 自動從 FaceForensics++ mini 切換到 Celeb-DF-v2 完整集
- 輸出結果到 `evaluation/results/scores_visual_ucf_celebdfv2.csv`

### 2.2 運行預設的 FaceForensics++ mini 評估（比較）

```bash
cd evaluation/run
python run_ucf_inference.py
# 或顯式指定
python run_ucf_inference.py mini
```

## 步驟 3：查看評估結果

評估完成後，檢查輸出 CSV：

```bash
# Celeb-DF-v2 結果
head evaluation/results/scores_visual_ucf_celebdfv2.csv

# FaceForensics++ mini 結果（用於比較）
head evaluation/results/scores_visual_ucf_v0.csv
```

### 結果格式
每行包含：
- `sample_id` — 樣本 ID
- `dataset` — 數據集名稱
- `label` — 真實標籤（0=真實, 1=偽造）
- `detector_name` — 檢測器名稱（UCF）
- `modality` — 模式（RGB）
- `fake_score` — 模型預測為偽造的概率（0-1，越高越可能是偽造）
- `score_type` — 分數類型（AUC）
- `inference_time_ms` — 推理時間（毫秒）
- `status` — 處理狀態（success/error）
- `error_message` — 錯誤信息（如有）

## 步驟 4：計算評估指標

### 4.1 計算 AUC（曲線下面積）

運行評估後，AUC 會自動在終端輸出。若要手動計算：

```python
import pandas as pd
from sklearn.metrics import roc_auc_score

# 加載評估結果
df = pd.read_csv('scores_visual_ucf_celebdfv2.csv')

# 計算 AUC
auc = roc_auc_score(df['label'], df['fake_score'])
print(f"AUC Score: {auc:.4f}")
```

### 4.2 其他常用指標

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 轉換為二分類預測（0.5 閾值）
y_pred = (df['fake_score'] >= 0.5).astype(int)

accuracy = accuracy_score(df['label'], y_pred)
precision = precision_score(df['label'], y_pred)
recall = recall_score(df['label'], y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
```

## 元數據文件結構

### mini_eval_metadata.csv（現有 - FaceForensics++）
- 50 個樣本（25 真實 + 25 偽造）
- 用於快速測試

### celebdfv2_metadata.csv（新建 - Celeb-DF-v2）
- 完整 Celeb-DF-v2 數據集
- Celeb-real 子集：所有真實視頻樣本
- Celeb-synthesis 子集：所有合成偽造視頻樣本

## 故障排除

### 問題：找不到 celebdfv2_metadata.csv
**解決方案：** 先運行 `python ../metadata/generate_celebdfv2_metadata.py`

### 問題：路徑錯誤或幀丟失
**解決方案：** 檢查 `datasets/rgb/Celeb-DF-v2/` 目錄結構是否完整

### 問題：CUDA 內存不足
**解決方案：** 減少批次大小或使用 CPU（設置 `DEVICE = "cpu"`）

### 問題：推理速度慢
**解決方案：** 確保使用 GPU（檢查 `DEVICE` 輸出應為 "cuda"）

## 修改評估配置

### 修改抽取幀數
編輯 `run_ucf_inference.py`：
```python
FRAMES_PER_CLIP = 8  # 改為 8 幀（默認 4）
```

### 使用不同的檢測器
按相同方式修改 `run_xception_inference.py` 或其他檢測器腳本

## 進階：自定義元數據

若要添加或修改評估樣本，直接編輯 CSV 文件：
- 添加新行用於新樣本
- `path` 列應包含相對於 REPO_ROOT 的幀目錄路徑
- `label` 應為 0（真實）或 1（偽造）

## 性能基準

期望值（UCF 檢測器在 Celeb-DF-v2 上）：
- AUC：~0.95-0.99（高性能）
- 推理速度：~100-300ms/樣本（取決於硬件）

## 參考命令快速總結

```bash
# 1. 生成元數據
cd evaluation && python generate_celebdfv2_metadata.py

# 2. 運行評估
python run_ucf_inference.py celebdfv2

# 3. 查看結果
head scores_visual_ucf_celebdfv2.csv

# 4. 或運行 Xception 基線
python run_xception_inference.py celebdfv2
```

---
更新日期：2024年
評估框架版本：v2（支持多數據集）
