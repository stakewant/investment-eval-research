import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models" / "bert_factor_labeler"

RANDOM_STATE = 42


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
    y = np.zeros((len(df), len(factor_ids)), dtype=np.float32)

    for row_idx, value in enumerate(df["gold_factor_ids"]):
        labels = split_items(value)
        for label in labels:
            if label in factor_to_idx:
                y[row_idx, factor_to_idx[label]] = 1.0

    return y


class RationaleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32)
        }

        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)

        return item


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


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    all_logits = []
    all_labels = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        labels = batch["labels"].cpu().numpy()
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits.detach().cpu().numpy()

        all_logits.append(logits)
        all_labels.append(labels)

    logits = np.vstack(all_logits)
    labels = np.vstack(all_labels)

    proba = 1 / (1 + np.exp(-logits))

    return labels, proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="klue/bert-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--freeze-encoder", action="store_true")
    args = parser.parse_args()

    rationale_path = DATA_DIR / "rationale_labels.csv"

    if not rationale_path.exists():
        raise FileNotFoundError("data/rationale_labels.csv 파일이 없습니다.")

    active = load_active_factors()
    factor_ids = get_all_factor_ids(active)

    df = pd.read_csv(rationale_path, encoding="utf-8-sig")

    required_cols = {"rationale_id", "answer_id", "rationale_text", "gold_factor_ids"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"rationale_labels.csv에 필요한 컬럼이 없습니다: {missing_cols}")

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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(factor_ids),
        problem_type="multi_label_classification",
        id2label={i: label for i, label in enumerate(factor_ids)},
        label2id={label: i for i, label in enumerate(factor_ids)}
    )

    if args.freeze_encoder:
        print("[INFO] Freeze encoder parameters")
        for param in model.base_model.parameters():
            param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = RationaleDataset(train_texts, y_train, tokenizer, args.max_length)
    test_dataset = RationaleDataset(test_texts, y_test, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr
    )

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"[Epoch {epoch}] train_loss={loss:.4f}")

    y_true, proba = predict(model, test_loader, device)

    y_pred = (proba >= args.threshold).astype(int)

    for i in range(len(y_pred)):
        if y_pred[i].sum() == 0:
            top1 = int(np.argmax(proba[i]))
            y_pred[i, top1] = 1

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    hit1 = hit_at_k(y_true, proba, 1)
    hit3 = hit_at_k(y_true, proba, 3)
    hit5 = hit_at_k(y_true, proba, 5)

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
        {"metric": "model_name", "score": args.model_name},
        {"metric": "sample_count", "score": len(df)},
        {"metric": "test_count", "score": len(test_idx)},
        {"metric": "label_count", "score": len(factor_ids)},
        {"metric": "Hit@1", "score": round(hit1, 4)},
        {"metric": "Hit@3", "score": round(hit3, 4)},
        {"metric": "Hit@5", "score": round(hit5, 4)},
        {"metric": "Micro-F1", "score": round(float(micro_f1), 4)},
        {"metric": "Macro-F1", "score": round(float(macro_f1), 4)}
    ])

    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    pred_path = OUTPUT_DIR / "bert_factor_predictions.csv"
    report_path = OUTPUT_DIR / "bert_factor_report.csv"
    factor_path = MODEL_DIR / "factor_ids.json"

    predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    with open(factor_path, "w", encoding="utf-8") as f:
        json.dump(factor_ids, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved predictions: {pred_path}")
    print(f"[OK] Saved report: {report_path}")
    print(f"[OK] Saved model: {MODEL_DIR}")

    print("\n=== BERT Factor Labeler Report ===")
    print(report)


if __name__ == "__main__":
    main()