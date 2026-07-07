import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command):
    print(f"\n[RUN] {command}")

    result = subprocess.run(
        command,
        shell=True,
        cwd=ROOT
    )

    if result.returncode != 0:
        print(f"\n[FAIL] {command}")
        sys.exit(result.returncode)

    print(f"[OK] {command}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["mlp", "bert", "all"],
        default="mlp",
        help="실행할 Phase 2 모델 선택"
    )
    args = parser.parse_args()

    if args.mode in ["mlp", "all"]:
        run("python src/train_neural_factor_labeler.py")

    if args.mode in ["bert", "all"]:
        run("python src/train_bert_factor_labeler.py")

    print(f"\n[DONE] Phase 2 completed: mode={args.mode}")


if __name__ == "__main__":
    main()