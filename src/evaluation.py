"""Held-out binary and category-wise evaluation; labels never enter the model."""
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def evaluate(scored):
    if "label" not in scored:
        return {"available": False, "reason": "No label column in testing data"}
    true = scored["label"].astype(int)
    predicted = (scored["anomaly_prediction"] == -1).astype(int)
    tn, fp, fn, tp = confusion_matrix(true, predicted, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        true, predicted, average="binary", zero_division=0)
    result = {"available": True, "tp": int(tp), "fp": int(fp), "tn": int(tn),
              "fn": int(fn), "precision": float(precision), "recall": float(recall),
              "f1": float(f1), "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}
    if "attack_cat" in scored:
        categories = []
        for category, group in scored.groupby("attack_cat", dropna=False):
            attack_rows = int((group["label"] == 1).sum())
            flagged = int((group["anomaly_prediction"] == -1).sum())
            detected = int(((group["label"] == 1) & (group["anomaly_prediction"] == -1)).sum())
            categories.append({"attack_cat": str(category), "total": len(group),
                               "attack_rows": attack_rows, "flagged": flagged,
                               "detected_attacks": detected,
                               "attack_recall": detected / attack_rows if attack_rows else None})
        result["per_category"] = categories
    return result
