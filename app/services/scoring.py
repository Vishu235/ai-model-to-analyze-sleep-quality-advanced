import numpy as np
from app.config import (
    EPOCH_SECONDS,
    SE_MIN, SE_GOOD, WASO_BAD_MIN, LATENCY_REF_MIN,
    N3_TARGET, REM_TARGET, FRAG_BAD_REF,
    W_SE, W_WASO, W_LATENCY, W_REST, W_FRAG,
)
from app.models.schemas import Metrics


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def compute_metrics(stages: list, confidence: list) -> Metrics:
    n = len(stages)
    recording_minutes = (n * EPOCH_SECONDS) / 60.0

    sleep_epochs = [s for s in stages if s != 0]
    tst_min = len(sleep_epochs) * EPOCH_SECONDS / 60.0
    sleep_efficiency = tst_min / recording_minutes if recording_minutes > 0 else 0.0

    first_sleep = next((i for i, s in enumerate(stages) if s != 0), None)
    sleep_latency_min = (
        first_sleep * EPOCH_SECONDS / 60.0 if first_sleep is not None else recording_minutes
    )

    waso_epochs = 0
    if first_sleep is not None:
        for s in stages[first_sleep:]:
            if s == 0:
                waso_epochs += 1
    waso_min = waso_epochs * EPOCH_SECONDS / 60.0

    pct_n3  = stages.count(3) / len(sleep_epochs) if sleep_epochs else 0.0
    pct_rem = stages.count(4) / len(sleep_epochs) if sleep_epochs else 0.0

    transitions = sum(1 for i in range(1, n) if stages[i] != stages[i - 1])
    fragmentation_index = transitions / n if n > 0 else 0.0
    confidence_mean = float(np.mean(confidence)) if confidence else 0.0

    se_score   = _clip01((sleep_efficiency - SE_MIN) / (SE_GOOD - SE_MIN)) * 100
    waso_score = _clip01((WASO_BAD_MIN - waso_min) / WASO_BAD_MIN) * 100
    lat_score  = _clip01((LATENCY_REF_MIN - sleep_latency_min) / LATENCY_REF_MIN) * 100
    n3_score   = _clip01(pct_n3  / N3_TARGET)  * 100
    rem_score  = _clip01(pct_rem / REM_TARGET)  * 100
    rest_score = 0.5 * (n3_score + rem_score)
    frag_score = _clip01((FRAG_BAD_REF - fragmentation_index) / FRAG_BAD_REF) * 100

    quality = (
        W_SE      * se_score
        + W_WASO  * waso_score
        + W_LATENCY * lat_score
        + W_REST  * rest_score
        + W_FRAG  * frag_score
    )

    return Metrics(
        recording_minutes   = round(recording_minutes, 2),
        tst_min             = round(tst_min, 2),
        sleep_latency_min   = round(sleep_latency_min, 2),
        sleep_efficiency    = round(sleep_efficiency, 4),
        waso_min            = round(waso_min, 2),
        pct_n3              = round(pct_n3, 4),
        pct_rem             = round(pct_rem, 4),
        fragmentation_index = round(fragmentation_index, 4),
        confidence_mean     = round(confidence_mean, 4),
        sleep_quality_score = round(quality, 2),
    )


def rule_based_explain(m: Metrics, label: str):
    """Fallback explanation when ANTHROPIC_API_KEY is not set."""
    parts = []
    if m.sleep_efficiency < 0.75:
        parts.append(
            f"Your sleep efficiency is low at {m.sleep_efficiency*100:.0f}% (target >85%), "
            "meaning much of your time in bed was spent awake."
        )
    else:
        parts.append(
            f"Your sleep efficiency is {m.sleep_efficiency*100:.0f}%, "
            f"which is {'good' if m.sleep_efficiency >= 0.85 else 'acceptable'}."
        )
    if m.waso_min > 30:
        parts.append(f"You woke up frequently after falling asleep, losing {m.waso_min:.0f} minutes of sleep.")
    if m.pct_n3 < 0.10:
        parts.append("Your deep sleep (N3) is below normal, which may affect physical recovery.")
    if m.pct_rem < 0.15:
        parts.append("Your REM sleep is low, which can impact memory and mood.")
    explanation = (
        " ".join(parts) if parts
        else f"Your sleep quality is {label.lower()} with a score of {m.sleep_quality_score:.0f}/100."
    )

    tips = []
    if m.sleep_latency_min > 20:
        tips.append("Maintain a consistent sleep schedule — same bedtime and wake time every day.")
    if m.waso_min > 30:
        tips.append("Avoid screens and bright light 1 hour before bed to reduce nighttime waking.")
    if m.pct_n3 < 0.15:
        tips.append("Exercise regularly (not within 3 hours of bedtime) to increase deep sleep.")
    if m.pct_rem < 0.20:
        tips.append("Limit alcohol — even small amounts significantly reduce REM sleep.")
    if m.fragmentation_index > 0.10:
        tips.append("Keep your bedroom cool (18–20°C) and dark to reduce sleep fragmentation.")
    defaults = [
        "Avoid caffeine after 2 PM.",
        "Use your bed only for sleep — not for work or screens.",
        "Try 10 minutes of breathing exercises before bed.",
    ]
    for d in defaults:
        if len(tips) >= 3:
            break
        if d not in tips:
            tips.append(d)
    return explanation, tips[:3]


# ---------------------------------------------------------------------------
# Private aliases kept for backward compatibility with existing tests
# ---------------------------------------------------------------------------
_compute_metrics    = compute_metrics
_rule_based_explain = rule_based_explain