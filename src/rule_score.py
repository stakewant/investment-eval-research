import json
from pathlib import Path

import pandas as pd
from io_utils import read_table

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

def load_action_policy():
    policy_path = DATA_DIR / "action_policy_s001.json"

    if not policy_path.exists():
        return {
            "factor_impacts": {},
            "action_scores": {}
        }

    with policy_path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)

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


def score_m2_factual_consistency(tags, mentioned=None):
    """
    M2. 정보 해석 정확도

    기존에는 fact_distortion, misinterpreted_direction, unsupported_claim 같은
    직접 오류 태그만 감점했기 때문에, 해석 자체가 거의 없는 답변도 5점이 나오는 문제가 있었다.

    수정 후에는 다음을 함께 반영한다.
    - 언급 요인 수
    - 근거 없는 주장 여부
    - 방향성 오해 / 사실 왜곡 여부
    - 필수 요인 누락으로 인한 해석 단순화
    - 원인-결과 연결 부족
    """
    if mentioned is None:
        mentioned = []

    tag_set = set(tags)
    mentioned_count = len(set(mentioned))

    # 1차 기준: 해석할 정보 자체가 얼마나 있는가
    if mentioned_count == 0:
        # 근거 요인이 없으면 정보 해석 정확도를 높게 줄 수 없음
        if (
            "unsupported_claim" in tag_set
            or "fact_distortion" in tag_set
            or "misinterpreted_direction" in tag_set
        ):
            return 1
        return 2

    if mentioned_count == 1:
        score = 3
    elif mentioned_count == 2:
        score = 4
    else:
        score = 5

    # 2차 기준: 명확한 해석 오류 감점
    if "fact_distortion" in tag_set:
        score -= 3

    if "misinterpreted_direction" in tag_set:
        score -= 2

    if "unsupported_claim" in tag_set:
        if mentioned_count <= 1:
            score -= 2
        else:
            score -= 1

    # 3차 기준: 핵심 맥락 부족으로 인한 해석 단순화 감점
    weak_context_tags = {
        "missing_required_factor",
        "ignored_mitigating_factor",
        "cause_effect_gap"
    }

    weak_context_count = len(tag_set & weak_context_tags)

    if mentioned_count == 2 and weak_context_count >= 1:
        score -= 1
    elif mentioned_count >= 3 and weak_context_count >= 2:
        score -= 2
    elif mentioned_count >= 3 and weak_context_count == 1:
        score -= 1

    return max(1, min(5, score))


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

def score_m4_action_alignment(tags, mentioned=None, action=None, action_policy=None):
    """
    M4. 행동-근거 정합성

    사건-영향 기준표를 이용해 사용자가 언급한 요인의 방향성과
    사용자의 행동 방향성이 일치하는지 평가한다.

    예:
    - 금융 불안, 유동성 위험, 외국인 매도 등 부정 요인 다수
      → 매수보다 관망/비중 축소/매도가 더 정합적
    - 정책 안전망, 직접 노출 제한, 재무 건전성 등 완화 요인 존재
      → 전량 매도는 과도할 수 있음
    """
    if mentioned is None:
        mentioned = []

    if action_policy is None:
        action_policy = {
            "factor_impacts": {},
            "action_scores": {}
        }

    tag_set = set(tags)
    mentioned_unique = list(set(mentioned))
    mentioned_count = len(mentioned_unique)

    factor_impacts = action_policy.get("factor_impacts", {})
    action_scores = action_policy.get("action_scores", {})

    # 근거 요인이 없으면 행동 정합성을 높게 줄 수 없음
    if mentioned_count == 0:
        score = 2
    else:
        impact_values = [
            float(factor_impacts[factor])
            for factor in mentioned_unique
            if factor in factor_impacts
        ]

        if not impact_values:
            score = 3
        else:
            impact_score = sum(impact_values)

            # 지나치게 큰 값을 완화
            if impact_score <= -4:
                expected_action = -2.0
            elif impact_score <= -2:
                expected_action = -1.0
            elif impact_score < 1:
                expected_action = 0.0
            elif impact_score < 3:
                expected_action = 0.5
            else:
                expected_action = 1.0

            action_score = float(action_scores.get(str(action), 0.0))
            gap = abs(expected_action - action_score)

            if gap <= 0.5:
                score = 5
            elif gap <= 1.0:
                score = 4
            elif gap <= 2.0:
                score = 3
            elif gap <= 3.0:
                score = 2
            else:
                score = 1

    # 직접적인 행동 문제 태그
    if "action_mismatch" in tag_set:
        score -= 2

    if "excessive_action" in tag_set:
        score -= 1

    if "passive_action" in tag_set:
        score -= 1

    if "missing_position_size" in tag_set:
        score -= 1

    # 근거 자체가 부실한 경우
    if "unsupported_claim" in tag_set:
        score -= 1

    if "conclusion_jump" in tag_set:
        score -= 1

    if "cause_effect_gap" in tag_set:
        score -= 1

    if "missing_required_factor" in tag_set and mentioned_count <= 1:
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


def compute_rule_scores(labels):
    active = load_active_factors()
    action_policy = load_action_policy()

    rows = []

    for _, row in labels.iterrows():
        mentioned = split_items(row.get("mentioned_factors", ""))
        tags = split_items(row.get("diagnostic_tags", ""))

        m1 = score_m1_key_factor_identification(mentioned, active)
        m2 = score_m2_factual_consistency(tags, mentioned)
        m3 = score_m3_risk_awareness(mentioned, tags, active)

        action = row.get("action", "")
        m4 = score_m4_action_alignment(tags, mentioned, action, action_policy)

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
            "mentioned_factors": row.get("mentioned_factors", ""),
            "diagnostic_tags": row.get("diagnostic_tags", "")
        })

    return pd.DataFrame(rows)

def main():
    labels = read_table(DATA_DIR / "human_labels.csv")

    answers_path = DATA_DIR / "answers.csv"
    if answers_path.exists():
        answers = read_table(answers_path)

        labels = labels.merge(
            answers[["answer_id", "action"]],
            on="answer_id",
            how="left"
        )
    else:
        labels["action"] = ""

    system_scores = compute_rule_scores(labels)

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "system_scores.csv"
    system_scores.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved system scores: {output_path}")
    print(system_scores)

if __name__ == "__main__":
    main()