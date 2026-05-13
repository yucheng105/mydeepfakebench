# Evaluation Manual

## 步驟 1：生成 Celeb-DF-v2 元數據 CSV

首先，列舉 Celeb-DF-v2 目錄結構並生成評估所需的元數據文件：

```bash
python evaluation/metadata/generate_celebdfv2_metadata.py
```

**輸出：**
- `evaluation/metadata/celebdfv2_metadata.csv` — 包含所有 Celeb-DF-v2 樣本的元數據
  - Celeb-real（真實視頻）樣本標籤為 0
  - Celeb-synthesis（合成視頻）樣本標籤為 1
  - 自動計算的幀數和時長信息

**如果已經有 MetaData 就不需要做這個了**

## 步驟 2：開始 Evaluate
### UCF
- **Within Domain**：FaceForensics++
```bash
python evaluation/run/run_ucf_inference.py
```
- **Cross Domain**：CelebDF-V2
```bash
python evaluation/run/run_ucf_inference.py celebdf-v2
```
