"""Train on UNSW-NB15 train split and score the held-out test split."""
import argparse
import json
from pathlib import Path
import joblib

from src.ingestion import load_flows
from src.engineering import engineer, feature_columns, matrix
from src.model import build_model, score
from src.evaluation import evaluate


def run(train_path, test_path, output_dir="outputs", normal_only=False, contamination=0.01, export_csv=False):
    train, train_stats = load_flows(train_path)
    test, test_stats = load_flows(test_path)
    if train.empty or test.empty:
        raise ValueError("Training and test splits must have retained rows")
    train, test = engineer(train), engineer(test)
    numeric, categorical = feature_columns(train)
    if normal_only:
        if "label" not in train:
            raise ValueError("--normal-only requires training labels")
        train_fit = train.loc[train["label"] == 0].copy()
        if train_fit.empty:
            raise ValueError("No normal training rows remain")
    else:
        train_fit = train
    model = build_model(numeric, categorical, contamination)
    model.fit(matrix(train_fit, numeric, categorical))
    prediction, anomaly_score = score(model, matrix(test, numeric, categorical))
    test["anomaly_prediction"] = prediction
    test["anomaly_score"] = anomaly_score
    test = test.sort_values("anomaly_score", ascending=False, kind="stable").reset_index(drop=True)
    flagged = test[test["anomaly_prediction"] == -1]
    # id order is an ordering proxy, not a timestamp or a causal sequence.
    first_id = flagged["id"].min() if not flagged.empty else None
    results = {
        "training_mode": "normal_only_label_selected" if normal_only else "unlabeled_all_training_rows",
        "train_cleaning": train_stats, "test_cleaning": test_stats,
        "training_rows_used": len(train_fit), "test_rows_scored": len(test),
        "selected_numeric_features": numeric, "selected_categorical_features": categorical,
        "contamination": contamination, "flagged_count": len(flagged),
        "flagged_outbound_bytes": int(flagged["sbytes"].sum()),
        "first_flagged_id_by_id_order": first_id.item() if hasattr(first_id, "item") else first_id,
        "evaluation": evaluate(test),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "numeric": numeric, "categorical": categorical}, output / "model.joblib")
    if export_csv:
        test.to_csv(output / "processed_scored_logs.csv", index=False)
    else:
        test.to_parquet(output / "processed_scored_logs.parquet", index=False)
    (output / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Processed UNSW-NB15 training .csv or .parquet")
    parser.add_argument("--test", required=True, help="Processed UNSW-NB15 testing .csv or .parquet")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--normal-only", action="store_true", help="Select label=0 training rows (uses labels for selection)")
    parser.add_argument("--contamination", type=float, default=0.01)
    parser.add_argument("--csv", action="store_true", help="Export scored CSV instead of Parquet")
    args = parser.parse_args()
    print(json.dumps(run(args.train, args.test, args.output_dir, args.normal_only,
                         args.contamination, args.csv), indent=2))
