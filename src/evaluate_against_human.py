from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


METRICS = ["M1", "M2", "M3", "M4", "M5"]


def evaluate():
    human = pd.read_csv(DATA_DIR / "human_labels.csv")
    system = pd.read_csv(OUTPUT_DIR / "system_scores.csv")

    # 사람 점수 평균
    human_avg = (
        human.groupby("answer_id")[METRICS]
        .mean()
        .reset_index()
        .rename(columns={m: f"human_{m}" for m in METRICS})
    )
    human_avg["human_final"] = human_avg[[f"human_{m}" for m in METRICS]].mean(axis=1)

    # 시스템 점수 평균
    system_cols = [f"system_{m}" for m in METRICS]
    system_avg = (
        system.groupby("answer_id")[system_cols]
        .mean()
        .reset_index()
    )
    system_avg["system_final"] = system_avg[system_cols].mean(axis=1)

    merged = pd.merge(human_avg, system_avg, on="answer_id", how="inner")

    rows = []

    for m in METRICS:
        human_col = f"human_{m}"
        system_col = f"system_{m}"

        mae = mean_absolute_error(merged[human_col], merged[system_col])
        corr, p_value = spearmanr(merged[human_col], merged[system_col])

        rows.append({
            "metric": m,
            "MAE": round(mae, 4),
            "Spearman": round(corr, 4) if not np.isnan(corr) else None,
            "p_value": round(p_value, 4) if not np.isnan(p_value) else None
        })

    final_mae = mean_absolute_error(merged["human_final"], merged["system_final"])
    final_corr, final_p = spearmanr(merged["human_final"], merged["system_final"])

    rows.append({
        "metric": "Final",
        "MAE": round(final_mae, 4),
        "Spearman": round(final_corr, 4) if not np.isnan(final_corr) else None,
        "p_value": round(final_p, 4) if not np.isnan(final_p) else None
    })

    report = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(exist_ok=True)
    merged_path = OUTPUT_DIR / "score_comparison.csv"
    report_path = OUTPUT_DIR / "score_eval_report.csv"

    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved comparison: {merged_path}")
    print(f"[OK] Saved report: {report_path}")
    print("\n=== Evaluation Report ===")
    print(report)


if __name__ == "__main__":
    evaluate()