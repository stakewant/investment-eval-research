from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


def split_items(value):
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def hit_at_k(gold_items, pred_items):
    gold_set = set(gold_items)
    pred_set = set(pred_items)
    return int(len(gold_set & pred_set) > 0)


def main():
    candidate_path = OUTPUT_DIR / "factor_candidates.csv"

    if not candidate_path.exists():
        raise FileNotFoundError(
            "outputs/factor_candidates.csv 파일이 없습니다. 먼저 retrieve_factor_candidates.py를 실행하세요."
        )

    df = pd.read_csv(candidate_path, encoding="utf-8-sig")

    rows = []

    top1_hits = []
    top3_hits = []
    top5_hits = []

    for _, row in df.iterrows():
        gold = split_items(row["gold_factor_ids"])
        top1 = split_items(row["top1"])
        top3 = split_items(row["top3"])
        top5 = split_items(row["top5"])

        h1 = hit_at_k(gold, top1)
        h3 = hit_at_k(gold, top3)
        h5 = hit_at_k(gold, top5)

        top1_hits.append(h1)
        top3_hits.append(h3)
        top5_hits.append(h5)

        rows.append({
            "rationale_id": row["rationale_id"],
            "gold_factor_ids": row["gold_factor_ids"],
            "top1": row["top1"],
            "top3": row["top3"],
            "top5": row["top5"],
            "hit@1": h1,
            "hit@3": h3,
            "hit@5": h5
        })

    detail = pd.DataFrame(rows)

    report = pd.DataFrame([
        {
            "metric": "Hit@1",
            "score": round(sum(top1_hits) / len(top1_hits), 4)
        },
        {
            "metric": "Hit@3",
            "score": round(sum(top3_hits) / len(top3_hits), 4)
        },
        {
            "metric": "Hit@5",
            "score": round(sum(top5_hits) / len(top5_hits), 4)
        }
    ])

    detail_path = OUTPUT_DIR / "factor_retrieval_detail.csv"
    report_path = OUTPUT_DIR / "factor_retrieval_report.csv"

    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved detail: {detail_path}")
    print(f"[OK] Saved report: {report_path}")
    print("\n=== Factor Retrieval Report ===")
    print(report)


if __name__ == "__main__":
    main()