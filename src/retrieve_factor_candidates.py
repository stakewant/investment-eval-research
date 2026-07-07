from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


TOP_K = 5


def build_factor_text(row):
    parts = [
        str(row["factor_id"]),
        str(row["factor_name"]),
        str(row["category"]),
        str(row["definition"]),
        str(row["representative_expressions"]),
    ]
    return " ".join(parts)


def main():
    factor_path = DATA_DIR / "factor_dictionary_s001.csv"
    rationale_path = DATA_DIR / "rationale_labels.csv"

    if not factor_path.exists():
        raise FileNotFoundError(f"Missing file: {factor_path}")

    if not rationale_path.exists():
        raise FileNotFoundError(f"Missing file: {rationale_path}")

    factors = pd.read_csv(factor_path, encoding="utf-8-sig")
    rationales = pd.read_csv(rationale_path, encoding="utf-8-sig")

    factors["factor_text"] = factors.apply(build_factor_text, axis=1)

    factor_texts = factors["factor_text"].tolist()
    rationale_texts = rationales["rationale_text"].astype(str).tolist()

    # 한국어는 단어 토큰화보다 문자 n-gram이 초기 baseline으로 안정적임
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1
    )

    all_texts = factor_texts + rationale_texts
    matrix = vectorizer.fit_transform(all_texts)

    factor_matrix = matrix[:len(factor_texts)]
    rationale_matrix = matrix[len(factor_texts):]

    sim = cosine_similarity(rationale_matrix, factor_matrix)

    rows = []

    for i, rationale_row in rationales.iterrows():
        scores = sim[i]
        ranked_idx = scores.argsort()[::-1][:TOP_K]

        candidates = []
        candidate_scores = []

        for idx in ranked_idx:
            candidates.append(factors.iloc[idx]["factor_id"])
            candidate_scores.append(round(float(scores[idx]), 4))

        rows.append({
            "rationale_id": rationale_row["rationale_id"],
            "answer_id": rationale_row["answer_id"],
            "rationale_text": rationale_row["rationale_text"],
            "gold_factor_ids": rationale_row["gold_factor_ids"],
            "top1": candidates[0],
            "top3": ";".join(candidates[:3]),
            "top5": ";".join(candidates[:5]),
            "top5_scores": ";".join(map(str, candidate_scores))
        })

    result = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "factor_candidates.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved factor candidates: {output_path}")
    print(result[["rationale_id", "top1", "top3"]])


if __name__ == "__main__":
    main()