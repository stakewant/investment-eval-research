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
    run("python src/validate_data.py")
    run("python src/rule_score.py")
    run("python src/evaluate_against_human.py")
    run("python src/make_tables.py")
    run("python src/analyze_score_errors.py")

    print("\n[DONE] 전체 평가 파이프라인 실행 완료")


if __name__ == "__main__":
    main()