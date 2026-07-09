from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

RANDOM_STATE = 42
THRESHOLD = 0.35


def split_items(value):
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def load_active_factors():
    path = DATA_DIR / "active_factors_s001.json"
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def get_all_factor_ids(active):
    return (
        active["required_factors"]
        + active["supporting_factors"]
        + active["portfolio_factors"]
    )


def build_label_matrix(df, factor_ids):
    factor_to_idx = {factor: i for i, factor in enumerate(factor_ids)}
    y = np.zeros((len(df), len(factor_ids)), dtype=int)

    for row_idx, value in enumerate(df["gold_factor_ids"]):
        labels = split_items(value)
        for label in labels:
            if label in factor_to_idx:
                y[row_idx, factor_to_idx[label]] = 1

    return y


def hit_at_k(y_true, proba, k):
    hits = []

    for i in range(len(y_true)):
        gold_idx = set(np.where(y_true[i] == 1)[0])
        pred_idx = set(np.argsort(proba[i])[::-1][:k])
        hits.append(int(len(gold_idx & pred_idx) > 0))

    return float(np.mean(hits)) if hits else 0.0


def topk_labels(proba_row, factor_ids, k=5):
    idxs = np.argsort(proba_row)[::-1][:k]
    return [factor_ids[i] for i in idxs], [float(proba_row[i]) for i in idxs]


def main():
    rationale_path = DATA_DIR / "rational_labels.csv"

    if not rationale_path.exists():
        raise FileNotFoundError("data/rational_labels.csv 파일이 없습니다.")

    active = load_active_factors()
    factor_ids = get_all_factor_ids(active)

    df = pd.read_csv(rationale_path, encoding="utf-8-sig")

    if "rationale_text" not in df.columns or "gold_factor_ids" not in df.columns:
        raise ValueError("rationale_labels.csv에는 rationale_text, gold_factor_ids 컬럼이 필요합니다.")

    texts = df["rationale_text"].astype(str).tolist()
    y = build_label_matrix(df, factor_ids)

    if len(df) < 20:
        print("[WARNING] 데이터가 20개 미만입니다. 현재는 학습/평가를 같은 데이터에서 수행합니다.")
        train_idx = np.arange(len(df))
        test_idx = np.arange(len(df))
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=0.25,
            random_state=RANDOM_STATE
        )

    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=5000,
        min_df=1
    )

    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)

    # 학습 데이터에 양성 예시가 하나도 없는 요인은 학습에서 제외
    trainable_indices = []
    for i in range(y_train.shape[1]):
        positive_count = int(y_train[:, i].sum())
        if 0 < positive_count < len(y_train):
            trainable_indices.append(i)

    if not trainable_indices:
        raise ValueError("학습 가능한 라벨이 없습니다. gold_factor_ids를 확인하세요.")

    trainable_factor_ids = [factor_ids[i] for i in trainable_indices]
    print(f"[INFO] 전체 요인 수: {len(factor_ids)}")
    print(f"[INFO] 학습 가능한 요인 수: {len(trainable_factor_ids)}")
    print(f"[INFO] 학습 가능한 요인: {trainable_factor_ids}")

    y_train_sub = y_train[:, trainable_indices]

    clf = OneVsRestClassifier(
        MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size="auto",
            learning_rate_init=0.001,
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    )

    clf.fit(x_train, y_train_sub)

    proba_sub = clf.predict_proba(x_test)

    if isinstance(proba_sub, list):
        proba_sub = np.vstack([p[:, 1] for p in proba_sub]).T

    if proba_sub.ndim == 1:
        proba_sub = proba_sub.reshape(-1, 1)

    proba = np.zeros((len(test_idx), len(factor_ids)), dtype=float)
    for col_idx, factor_idx in enumerate(trainable_indices):
        proba[:, factor_idx] = proba_sub[:, col_idx]

    y_pred = (proba >= THRESHOLD).astype(int)

    # 아무 라벨도 예측되지 않은 샘플은 top1을 강제로 양성 처리
    for i in range(len(y_pred)):
        if y_pred[i].sum() == 0:
            top1 = int(np.argmax(proba[i]))
            y_pred[i, top1] = 1

    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    hit1 = hit_at_k(y_test, proba, 1)
    hit3 = hit_at_k(y_test, proba, 3)
    hit5 = hit_at_k(y_test, proba, 5)

    prediction_rows = []

    for local_i, original_idx in enumerate(test_idx):
        top_labels, top_scores = topk_labels(proba[local_i], factor_ids, k=5)

        prediction_rows.append({
            "rationale_id": df.iloc[original_idx]["rationale_id"],
            "answer_id": df.iloc[original_idx]["answer_id"],
            "rationale_text": df.iloc[original_idx]["rationale_text"],
            "gold_factor_ids": df.iloc[original_idx]["gold_factor_ids"],
            "pred_top1": top_labels[0],
            "pred_top3": ";".join(top_labels[:3]),
            "pred_top5": ";".join(top_labels),
            "pred_top5_scores": ";".join([str(round(s, 4)) for s in top_scores])
        })

    predictions = pd.DataFrame(prediction_rows)

    report = pd.DataFrame([
        {"metric": "sample_count", "score": len(df)},
        {"metric": "test_count", "score": len(test_idx)},
        {"metric": "trainable_label_count", "score": len(trainable_indices)},
        {"metric": "Hit@1", "score": round(hit1, 4)},
        {"metric": "Hit@3", "score": round(hit3, 4)},
        {"metric": "Hit@5", "score": round(hit5, 4)},
        {"metric": "Micro-F1", "score": round(float(micro_f1), 4)},
        {"metric": "Macro-F1", "score": round(float(macro_f1), 4)}
    ])

    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    pred_path = OUTPUT_DIR / "neural_factor_predictions.csv"
    report_path = OUTPUT_DIR / "neural_factor_report.csv"
    model_path = MODEL_DIR / "neural_factor_labeler.joblib"

    predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    joblib.dump({
        "vectorizer": vectorizer,
        "classifier": clf,
        "factor_ids": factor_ids,
        "trainable_indices": trainable_indices,
        "threshold": THRESHOLD
    }, model_path)

    print(f"[OK] Saved predictions: {pred_path}")
    print(f"[OK] Saved report: {report_path}")
    print(f"[OK] Saved model: {model_path}")

    print("\n=== Neural Factor Labeler Report ===")
    print(report)


if __name__ == "__main__":
    main()