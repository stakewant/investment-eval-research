import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "bert_factor_labeler"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, help="요인 라벨을 예측할 근거 문장")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not MODEL_DIR.exists():
        raise FileNotFoundError("models/bert_factor_labeler 폴더가 없습니다. 먼저 train_bert_factor_labeler.py를 실행하세요.")

    factor_path = MODEL_DIR / "factor_ids.json"
    if not factor_path.exists():
        raise FileNotFoundError("models/bert_factor_labeler/factor_ids.json 파일이 없습니다.")

    with open(factor_path, "r", encoding="utf-8") as f:
        factor_ids = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    encoded = tokenizer(
        args.text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    outputs = model(**encoded)
    logits = outputs.logits.detach().cpu().numpy()[0]
    proba = 1 / (1 + np.exp(-logits))

    ranked = np.argsort(proba)[::-1][:args.top_k]

    print("\n=== BERT Prediction Result ===")
    print(f"input: {args.text}")

    for rank, idx in enumerate(ranked, start=1):
        print(f"{rank}. {factor_ids[idx]}  score={proba[idx]:.4f}")


if __name__ == "__main__":
    main()