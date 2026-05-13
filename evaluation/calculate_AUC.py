from sklearn.metrics import roc_auc_score
import csv

y_true, y_score = [], []

#################################
# Enter File Name Here: CSV only
#################################
filename = "results/ucf/scores_visual_ucf_celebdfv2.csv"

with open(filename) as f:
    for row in csv.DictReader(f):
        if row["status"] == "ok":
            y_true.append(int(row["label"]))
            y_score.append(float(row["fake_score"]))

auc = roc_auc_score(y_true, y_score)
print(f"from {filename}:")
print(f"AUC: {auc:.4f}")