from sklearn.metrics import roc_auc_score
import csv

y_true, y_score = [], []

#################################
# Enter File Name Here: CSV only
#################################
filename = "results/xception_fine_tuned/scores_visual_xception_ft_celebdf_v0.csv"

with open(filename) as f:
    for row in csv.DictReader(f):
        if row["status"] == "ok":
            y_true.append(int(row["label"]))
            y_score.append(float(row["fake_score"]))

auc = roc_auc_score(y_true, y_score)
print(f"from {filename}:")
print(f"AUC: {auc:.4f}")