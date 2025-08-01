import os
import glob
import json, re
from pathlib import Path

from sklearn.metrics import (precision_recall_fscore_support, accuracy_score, mean_absolute_error)
GT_DIR   = Path("validation/receipt_val_data/ground_truth_json")
PRED_DIR = Path("validation/receipt_val_data/predictions")

y_true_names = []
y_pred_names = []
y_true_cats  = []
y_pred_cats  = []
y_true_ppu   = []
y_pred_ppu   = []
y_true_qty   = []
y_pred_qty   = []
y_true_tot   = []
y_pred_tot   = []

tokenizer = re.compile(r"\w+")
def tokenize(text):
    return tokenizer.findall(text.lower())

for gt_path in sorted(GT_DIR.glob("*.json")):
    stem      = gt_path.stem
    pred_path = PRED_DIR / f"{stem}.json"
    if not pred_path.exists():
        print(f"Skipping {stem}: no prediction file found.")
        continue
    with gt_path.open("r", encoding="utf-8") as f:
        gt_items = json.load(f)
    with pred_path.open("r", encoding="utf-8") as f:
        pred_items = json.load(f)
    n = min(len(gt_items), len(pred_items))
    for i in range(n):
        gt   = gt_items[i]
        pred = pred_items[i]
        y_true_names.append(gt["item"])
        y_pred_names.append(pred["name"])
        y_true_cats.append(gt["category"])
        y_pred_cats.append(pred["category"])
        y_true_ppu.append(gt["price_per_unit_item"])
        y_pred_ppu.append(pred["price_per_unit"])
        y_true_qty.append(gt["quantity"])
        y_pred_qty.append(pred["quantity"])
        y_true_tot.append(gt["total_price"])
        y_pred_tot.append(pred["total_price"])
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true_names, y_pred_names, average="micro"
)
cat_acc = accuracy_score(y_true_cats, y_pred_cats)
ppu_mae = mean_absolute_error(y_true_ppu, y_pred_ppu)
qty_mae = mean_absolute_error(y_true_qty, y_pred_qty)
tot_mae = mean_absolute_error(y_true_tot, y_pred_tot)
print("ITEM NAME   →  precision: {:.3f}, recall: {:.3f},  F1 (micro): {:.3f}".format(
    precision, recall, f1
))
print("CATEGORY    →  accuracy: {:.3f}".format(cat_acc))
print("PRICE/UNIT  →  MAE: {:.3f}".format(ppu_mae))
print("QUANTITY    →  MAE: {:.3f}".format(qty_mae))
print("TOTAL PRICE →  MAE: {:.3f}".format(tot_mae))
tp = fp = fn = 0
for gt_name, pred_name in zip(y_true_names, y_pred_names):
    gt_tokens   = set(tokenize(gt_name))
    pred_tokens = set(tokenize(pred_name))
    tp += len(gt_tokens & pred_tokens)
    fp += len(pred_tokens - gt_tokens)
    fn += len(gt_tokens - pred_tokens)
token_prec = tp / (tp + fp) if (tp + fp) else 0.0
token_rec  = tp / (tp + fn) if (tp + fn) else 0.0
token_f1   = (2 * token_prec * token_rec / (token_prec + token_rec)
              if (token_prec + token_rec) else 0.0)
print("TOKEN-LEVEL ITEM NAME →  precision: {:.3f}, recall: {:.3f}, F1: {:.3f}".format(
    token_prec, token_rec, token_f1
))