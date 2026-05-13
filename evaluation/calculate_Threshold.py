import csv
import numpy as np

rows = []
with open("results/ucf/scores_visual_ucf_ff++v0.csv") as f:
    rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]

y_true = np.array([int(r["label"]) for r in rows])
y_score = np.array([float(r["fake_score"]) for r in rows])
latency = np.mean([float(r["inference_time_ms"]) for r in rows])
n = len(rows)

sweep_rows = []
for t in np.arange(0.1, 1.0, 0.1):
    pred = (y_score >= t).astype(int)
    far = np.mean(pred[y_true == 0])       # 真實影片被判成假的比例
    frr = np.mean(1 - pred[y_true == 1])   # 假的影片沒被抓到的比例
    sweep_rows.append({
        "detector_name": "UCF",            # 換成你的 detector 名稱
        "threshold": round(t, 2),
        "far": round(float(far), 4),
        "frr": round(float(frr), 4),
        "uncertain_rate": 0.0,
        "avg_latency_ms": round(latency, 2),
        "n_samples": n,
    })

with open("threshold_sweep_ucf_v0.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=sweep_rows[0].keys())
    writer.writeheader()
    writer.writerows(sweep_rows)
