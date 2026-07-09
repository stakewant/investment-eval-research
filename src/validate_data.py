import json
from pathlib import Path

import pandas as pd
from io_utils import read_table

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

REQUIRED_COLUMNS = [
    "answer_id",
    "annotator",
    "mentioned_factors",
    "missing_required_factors",
    "diagnostic_tags",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "comment",
]


def split_items(value):
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def main():
    errors = []
    warnings = []

    active_path = DATA_DIR / "active_factors_s001.json"
    tags_path = DATA_DIR / "diagnostic_tags.json"
    labels_path = DATA_DIR / "human_labels.csv"

    if not active_path.exists():
        errors.append("data/active_factors_s001.json 파일이 없습니다.")

    if not labels_path.exists():
        errors.append("data/human_labels.csv 파일이 없습니다.")

    if errors:
        print("[ERROR] 필수 파일 누락")
        for e in errors:
            print("-", e)
        return

    active = load_json(active_path)

    all_factors = set(
        active["required_factors"]
        + active["supporting_factors"]
        + active["portfolio_factors"]
    )

    required_factors = set(active["required_factors"])

    valid_tags = set()
    if tags_path.exists():
        tag_data = load_json(tags_path)
        for tag_list in tag_data.values():
            valid_tags.update(tag_list)
    else:
        warnings.append("diagnostic_tags.json 파일이 없어 태그 검증은 건너뜀")

    labels = read_table(labels_path)

    for col in REQUIRED_COLUMNS:
        if col not in labels.columns:
            errors.append(f"human_labels.csv에 필요한 컬럼이 없습니다: {col}")

    if errors:
        print("[ERROR] 컬럼 오류")
        for e in errors:
            print("-", e)
        return

    for idx, row in labels.iterrows():
        row_num = idx + 2

        for metric in ["M1", "M2", "M3", "M4", "M5"]:
            try:
                score = int(row[metric])
                if score < 1 or score > 5:
                    errors.append(f"Row {row_num}: {metric} 점수는 1~5 사이여야 합니다. 현재 값: {row[metric]}")
            except Exception:
                errors.append(f"Row {row_num}: {metric} 점수가 숫자가 아닙니다. 현재 값: {row[metric]}")

        mentioned = split_items(row["mentioned_factors"])
        missing = split_items(row["missing_required_factors"])
        tags = split_items(row["diagnostic_tags"])

        for factor in mentioned:
            if factor not in all_factors:
                errors.append(f"Row {row_num}: 알 수 없는 mentioned_factor: {factor}")

        for factor in missing:
            if factor not in required_factors:
                errors.append(f"Row {row_num}: missing_required_factors에는 필수 요인만 입력해야 합니다: {factor}")

        if valid_tags:
            for tag in tags:
                if tag not in valid_tags:
                    errors.append(f"Row {row_num}: 알 수 없는 diagnostic_tag: {tag}")

    duplicated = labels.duplicated(subset=["answer_id", "annotator"], keep=False)
    if duplicated.any():
        dup = labels[duplicated][["answer_id", "annotator"]]
        errors.append(f"중복 라벨이 있습니다:\n{dup}")

    label_count = labels.groupby("answer_id")["annotator"].nunique()

    for answer_id, count in label_count.items():
        if count < 2:
            warnings.append(f"{answer_id}: 라벨러가 2명보다 적습니다. 현재 {count}명")
        elif count > 2:
            warnings.append(f"{answer_id}: 라벨러가 2명보다 많습니다. 현재 {count}명")

    print("\n=== Data Validation Result ===")

    if errors:
        print("\n[ERROR]")
        for e in errors:
            print("-", e)
    else:
        print("\n[OK] 치명적인 오류 없음")

    if warnings:
        print("\n[WARNING]")
        for w in warnings:
            print("-", w)
    else:
        print("\n[OK] 경고 없음")


if __name__ == "__main__":
    main()