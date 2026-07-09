import json
from pathlib import Path

import pandas as pd
from io_utils import read_table

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def split_items(value):
    """
    CSV 안에서 세미콜론으로 연결된 값을 리스트로 변환한다.
    빈 값은 빈 리스트로 처리한다.
    """
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def load_active_factors():
    with open(DATA_DIR / "active_factors_s001.json", "r", encoding="utf-8-sig") as f:
        return json.load(f)


def score_m1_key_factor_identification(mentioned, active):
    """
    M1. 핵심 정보 식별도
    필수 요인 5개 중 몇 개를 언급했는지 중심으로 계산한다.
    """
    required = set(active["required_factors"])
    supporting = set(active["supporting_factors"])
    portfolio = set(active["portfolio_factors"])

    mentioned_set = set(mentioned)

    required_count = len(required & mentioned_set)
    supporting_count = len(supporting & mentioned_set)
    portfolio_count = len(portfolio & mentioned_set)

    if required_count >= 4:
        return 5
    if required_count == 3:
        return 4
    if required_count == 2:
        return 3
    if required_count == 1:
        if supporting_count + portfolio_count >= 2:
            return 2
        return 2
    return 1


def score_m2_factual_consistency(tags):
    """
    M2. 정보 해석 정확도
    방향성 오해, 사실 왜곡, 근거 없는 주장이 있으면 감점한다.
    """
    score = 5

    if "fact_distortion" in tags:
        score -= 3
    if "misinterpreted_direction" in tags:
        score -= 2
    if "unsupported_claim" in tags:
        score -= 2

    return max(1, score)


def score_m3_risk_awareness(mentioned, tags, active):
    """
    M3. 위험 인식도
    위험 증가 요인과 완화 요인을 균형 있게 고려했는지 계산한다.
    """
    mentioned_set = set(mentioned)
    risk_factors = set(active["risk_increase_factors"])
    mitigating_factors = set(active["risk_mitigating_factors"])

    risk_count = len(risk_factors & mentioned_set)
    mitigating_count = len(mitigating_factors & mentioned_set)

    has_uncertainty = "EVENT_UNCERTAINTY" in mentioned_set

    if risk_count >= 3 and mitigating_count >= 2:
        score = 5
    elif risk_count >= 2 and mitigating_count >= 1:
        score = 4
    elif risk_count >= 1 and (mitigating_count >= 1 or has_uncertainty):
        score = 3
    elif risk_count >= 1:
        score = 2
    else:
        score = 1

    if "ignored_mitigating_factor" in tags:
        score -= 1
    if "ignored_uncertainty" in tags:
        score -= 1
    if "no_counter_scenario" in tags:
        score -= 1

    return max(1, min(5, score))


def score_m4_action_alignment(tags, mentioned=None):
    """
    M4. 행동-근거 정합성

    기존에는 행동 관련 진단 태그만 보고 감점했기 때문에,
    근거 요인이 거의 없거나 unsupported_claim/cause_effect_gap이 있는 답변도
    M4가 과도하게 높게 나오는 문제가 있었다.

    수정 후에는 다음을 함께 반영한다.
    - 언급 요인 수
    - 행동 관련 태그
    - 근거 부족 태그
    - 결론 비약 / 원인-결과 연결 부족
    """
    if mentioned is None:
        mentioned = []

    tag_set = set(tags)
    mentioned_count = len(set(mentioned))

    score = 5

    # 근거 요인이 없거나 거의 없는 경우 행동 정합성을 높게 줄 수 없음
    if mentioned_count == 0:
        score -= 3
    elif mentioned_count == 1:
        score -= 1

    # 직접적인 행동-근거 불일치
    if "action_mismatch" in tag_set:
        score -= 3

    if "excessive_action" in tag_set:
        score -= 2

    if "passive_action" in tag_set:
        score -= 2

    if "missing_position_size" in tag_set:
        score -= 1

    # 근거 자체가 약하면 행동 정합성도 낮아져야 함
    if "unsupported_claim" in tag_set:
        score -= 2

    if "conclusion_jump" in tag_set:
        score -= 2

    if "cause_effect_gap" in tag_set:
        score -= 1

    # 필수 요인이나 완화 요인을 거의 보지 않은 경우 행동 강도 판단도 불안정함
    if "missing_required_factor" in tag_set and mentioned_count <= 2:
        score -= 1

    if "ignored_mitigating_factor" in tag_set:
        score -= 1

    if "ignored_uncertainty" in tag_set:
        score -= 1

    return max(1, min(5, score))


def score_m5_logical_coherence(tags):
    """
    M5. 논리 일관성
    내부 모순, 결론 비약, 원인-결과 연결 부족을 감점한다.
    """
    score = 5

    if "internal_contradiction" in tags:
        score -= 3
    if "conclusion_jump" in tags:
        score -= 2
    if "cause_effect_gap" in tags:
        score -= 2
    if "unsupported_claim" in tags:
        score -= 1

    return max(1, score)


def compute_rule_scores():
    active = load_active_factors()
    labels = read_table(DATA_DIR / "human_labels.csv")

    rows = []

    for _, row in labels.iterrows():
        mentioned = split_items(row["mentioned_factors"])
        tags = split_items(row["diagnostic_tags"])

        m1 = score_m1_key_factor_identification(mentioned, active)
        m2 = score_m2_factual_consistency(tags)
        m3 = score_m3_risk_awareness(mentioned, tags, active)
        m4 = score_m4_action_alignment(tags, mentioned)
        m5 = score_m5_logical_coherence(tags)

        final_score = round((m1 + m2 + m3 + m4 + m5) / 5, 3)

        rows.append({
            "answer_id": row["answer_id"],
            "annotator": row["annotator"],
            "system_M1": m1,
            "system_M2": m2,
            "system_M3": m3,
            "system_M4": m4,
            "system_M5": m5,
            "system_final": final_score,
            "mentioned_factors": row["mentioned_factors"],
            "diagnostic_tags": row["diagnostic_tags"]
        })

    system_scores = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "system_scores.csv"
    system_scores.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved system scores: {output_path}")
    print(system_scores)


if __name__ == "__main__":
    compute_rule_scores()