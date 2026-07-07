import argparse
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "neural_factor_labeler.joblib"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, help="요인 라벨을 예측할 근거 문장")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError("models/neural_factor_labeler.joblib 파일이 없습니다. 먼저 train_neural_factor_labeler.py를 실행하세요.")

    bundle = joblib.load(MODEL_PATH)

    vectorizer = bundle["vectorizer"]
    clf = bundle["classifier"]
    factor_ids = bundle["factor_ids"]
    trainable_indices = bundle["trainable_indices"]

    x = vectorizer.transform([args.text])
    proba_sub = clf.predict_proba(x)

    if isinstance(proba_sub, list):
        proba_sub = np.vstack([p[:, 1] for p in proba_sub]).T

    if proba_sub.ndim == 1:
        proba_sub = proba_sub.reshape(1, -1)

    proba = np.zeros(len(factor_ids), dtype=float)

    for col_idx, factor_idx in enumerate(trainable_indices):
        proba[factor_idx] = proba_sub[0, col_idx]

    ranked = np.argsort(proba)[::-1][:args.top_k]

    print("\n=== Prediction Result ===")
    print(f"input: {args.text}")

    for rank, idx in enumerate(ranked, start=1):
        print(f"{rank}. {factor_ids[idx]}  score={proba[idx]:.4f}")


if __name__ == "__main__":
    main()