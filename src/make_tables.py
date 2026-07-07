from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


def make_tables():
    report = pd.read_csv(OUTPUT_DIR / "score_eval_report.csv")

    table = report.rename(columns={
        "metric": "Evaluation Dimension",
        "MAE": "Mean Absolute Error",
        "Spearman": "Spearman Correlation",
        "p_value": "p-value"
    })

    output_path = OUTPUT_DIR / "paper_table_scores.csv"
    table.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved paper table: {output_path}")
    print(table)


if __name__ == "__main__":
    make_tables()