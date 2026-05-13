# 如何用 run_ucf_inference.py 對 Celeb-DF-v2 進行評估

## 快速開始（3 步）

### 步驟 1：生成元數據 CSV
在 `evaluation` 資料夾執行：
```bash
cd evaluation
python generate_celebdfv2_metadata.py
```
✅ 生成 `celebdfv2_metadata.csv` — 自動掃描所有 Celeb-DF-v2 樣本

### 步驟 2：運行 UCF 檢測器推理
```bash
python run_ucf_inference.py celebdfv2
```
✅ 自動讀取 `celebdfv2_metadata.csv`  
✅ 對每個樣本提取 4 幀進行推理  
✅ 生成 `scores_visual_ucf_celebdfv2.csv` 結果

### 步驟 3：查看結果
```bash
head scores_visual_ucf_celebdfv2.csv
```
✅ 檢查 AUC 分數和推理時間

---

## 修改說明

### 文件修改清單

#### 1. **evaluation/generate_celebdfv2_metadata.py** （新建）
- **功能**：自動掃描 Celeb-DF-v2 目錄生成元數據
- **輸出**：`celebdfv2_metadata.csv` 含所有樣本的路徑和標籤
- **使用**：`python generate_celebdfv2_metadata.py`

#### 2. **evaluation/run_ucf_inference.py** （修改）
- **改變**：新增數據集選擇參數
- **用法**：
  ```
  python run_ucf_inference.py              # 默認：mini (FaceForensics++)
  python run_ucf_inference.py celebdfv2    # 新增：Celeb-DF-v2
  ```
- **邏輯**：根據第一個參數決定元數據路徑和輸出路徑

#### 3. **evaluation/run_xception_inference.py** （修改）
- **改變**：同步新增數據集選擇參數
- **用法**：
  ```
  python run_xception_inference.py celebdfv2   # Xception 基線評估
  ```

#### 4. **evaluation/CELEB_DF_V2_EVAL_GUIDE.md** （新建）
- 詳細的評估指南和故障排除

---

## 數據流程

```
datasets/rgb/Celeb-DF-v2/
├── Celeb-real/frames/
│   ├── id0_0000/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   ├── id0_0001/
│   └── ...
└── Celeb-synthesis/frames/
    ├── id0_id16_0000/
    │   ├── 0.jpg
    │   ├── 1.jpg
    │   └── ...
    └── ...
         ⬇
    generate_celebdfv2_metadata.py
         ⬇
    celebdfv2_metadata.csv
    (path, label, fps, duration_sec, etc.)
         ⬇
    run_ucf_inference.py celebdfv2
         ⬇
    scores_visual_ucf_celebdfv2.csv
    (sample_id, label, fake_score, AUC, etc.)
```

---

## 關鍵命令總結

| 任務 | 命令 |
|------|------|
| 生成 Celeb-DF-v2 元數據 | `cd evaluation && python generate_celebdfv2_metadata.py` |
| UCF 評估（Celeb-DF-v2） | `python run_ucf_inference.py celebdfv2` |
| Xception 評估（Celeb-DF-v2） | `python run_xception_inference.py celebdfv2` |
| UCF 評估（FaceForensics++ mini） | `python run_ucf_inference.py` 或 `python run_ucf_inference.py mini` |
| 查看 UCF 結果 | `head scores_visual_ucf_celebdfv2.csv` |
| 查看 Xception 結果 | `head scores_visual_xception_celebdfv2.csv` |

---

## 默認行為

### run_ucf_inference.py
- **無參數**：讀取 `mini_eval_metadata.csv`，輸出 `scores_visual_ucf_v0.csv`
- **參數 `celebdfv2`**：讀取 `celebdfv2_metadata.csv`，輸出 `scores_visual_ucf_celebdfv2.csv`
- **參數 `mini`**：同無參數（顯式指定）

### run_xception_inference.py
- **無參數**：FaceForensics++ mini 評估
- **參數 `celebdfv2`**：Celeb-DF-v2 評估

---

## 預期輸出 CSV 格式

### scores_visual_ucf_celebdfv2.csv
```
sample_id,dataset,label,detector_name,modality,fake_score,score_type,inference_time_ms,status,error_message
celeb_real_000,Celeb-DF-v2-Real,0,UCF,RGB,0.0123,AUC,256.34,success,
celeb_real_001,Celeb-DF-v2-Real,0,UCF,RGB,0.0456,AUC,243.12,success,
celeb_fake_000,Celeb-DF-v2-Synthesis,1,UCF,RGB,0.9876,AUC,251.23,success,
celeb_fake_001,Celeb-DF-v2-Synthesis,1,UCF,RGB,0.9654,AUC,248.87,success,
...
```

---

## 注意事項

1. **幀目錄路徑**：元數據中的 `path` 列應指向包含 `.jpg` 幀文件的目錄
2. **標籤規則**：`label=0` 為真實，`label=1` 為偽造/合成
3. **代碼向後兼容**：運行不帶參數仍使用原有的 mini 數據集
4. **自動路徑轉換**：支持將舊的 macOS 路徑 (`/Users/nia/...`) 自動轉換為當前工作區路徑

---

## 常見問題

**Q：執行 `python run_ucf_inference.py celebdfv2` 時出錯"celebdfv2_metadata.csv 不存在"**  
A：先運行 `python generate_celebdfv2_metadata.py` 生成元數據文件

**Q：可以一次評估多個數據集嗎？**  
A：可以，依次運行：
```bash
python run_ucf_inference.py           # FaceForensics++ mini
python run_ucf_inference.py celebdfv2 # Celeb-DF-v2
```

**Q：可以修改每個樣本提取的幀數嗎？**  
A：可以，編輯 `run_ucf_inference.py` 中的 `FRAMES_PER_CLIP = 4` 改為其他值

**Q：評估速度很慢，如何加速？**  
A：確保使用 GPU（檢查終端輸出中的 `DEVICE: cuda`），或減少樣本數進行測試

---

## 更新歷史

- **v1**（原始）：固定 FaceForensics++ mini 數據集
- **v2**（本次更新）：
  - 新增 Celeb-DF-v2 完整數據集支持
  - 新增 `generate_celebdfv2_metadata.py` 自動生成元數據
  - 修改 `run_ucf_inference.py` 和 `run_xception_inference.py` 支持數據集參數
  - 新增詳細的評估指南文檔

---

準備好了嗎？現在可以開始對 Celeb-DF-v2 進行完整評估了！🚀
