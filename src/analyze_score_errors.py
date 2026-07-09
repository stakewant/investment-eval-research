from pathlib import Path

import pandas as pd

from io_utils import read_table


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

METRICS = ["M1", "M2", "M3", "M4", "M5"]


def main():
    answers_path = DATA_DIR / "answers.csv"
    human_path = DATA_DIR / "human_labels.csv"
    system_path = OUTPUT_DIR / "system_scores.csv"

    human = read_table(human_path)
    system = read_table(system_path)

    if answers_path.exists():
        answers = read_table(answers_path)
    else:
        answers = pd.DataFrame(columns=["answer_id", "answer_text", "action", "answer_type"])

    human_avg = (
        human.groupby("answer_id")[METRICS]
        .mean()
        .reset_index()
        .rename(columns={m: f"human_{m}" for m in METRICS})
    )
    human_avg["human_final"] = human_avg[[f"human_{m}" for m in METRICS]].mean(axis=1)

    system_cols = [f"system_{m}" for m in METRICS]
    system_avg = (
        system.groupby("answer_id")[system_cols]
        .mean()
        .reset_index()
    )
    system_avg["system_final"] = system_avg[system_cols].mean(axis=1)

    merged = pd.merge(human_avg, system_avg, on="answer_id", how="inner")

    keep_answer_cols = [col for col in ["answer_id", "answer_type", "action", "answer_text"] if col in answers.columns]
    if keep_answer_cols:
        merged = pd.merge(merged, answers[keep_answer_cols], on="answer_id", how="left")

    for m in METRICS:
        merged[f"error_{m}"] = (merged[f"human_{m}"] - merged[f"system_{m}"]).abs()

    merged["error_final"] = (merged["human_final"] - merged["system_final"]).abs()

    error_cols = [f"error_{m}" for m in METRICS] + ["error_final"]
    score_cols = []
    for m in METRICS:
        score_cols.extend([f"human_{m}", f"system_{m}", f"error_{m}"])

    output_cols = (
        ["answer_id"]
        + [col for col in ["answer_type", "action"] if col in merged.columns]
        + score_cols
        + ["human_final", "system_final", "error_final"]
        + [col for col in ["answer_text"] if col in merged.columns]
    )

    detail = merged[output_cols].sort_values("error_final", ascending=False)

    top_m2 = merged.sort_values("error_M2", ascending=False).head(10)
    top_m4 = merged.sort_values("error_M4", ascending=False).head(10)

    OUTPUT_DIR.mkdir(exist_ok=True)

    detail_path = OUTPUT_DIR / "score_error_detail.csv"
    top_m2_path = OUTPUT_DIR / "top_m2_errors.csv"
    top_m4_path = OUTPUT_DIR / "top_m4_errors.csv"

    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    top_m2.to_csv(top_m2_path, index=False, encoding="utf-8-sig")
    top_m4.to_csv(top_m4_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved detail: {detail_path}")
    print(f"[OK] Saved M2 errors: {top_m2_path}")
    print(f"[OK] Saved M4 errors: {top_m4_path}")

    print("\n=== Top Final Score Errors ===")
    print(detail[["answer_id", "human_final", "system_final", "error_final"]].head(10))

    print("\n=== Top M4 Errors ===")
    print(top_m4[["answer_id", "human_M4", "system_M4", "error_M4"]].head(10))


if __name__ == "__main__":
    main()