import json

from fastapi import APIRouter, Depends

from app.auth import require_api_key
from app import state
from app.models.schemas import ExplainRequest, ExplainResponse
from app.services.scoring import rule_based_explain

router = APIRouter()


@router.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest, _=Depends(require_api_key)):
    """Claude Haiku explains sleep metrics in plain language with 3 actionable tips."""
    m = req.metrics
    score = m.sleep_quality_score

    if score >= 75:
        label = "Great"
    elif score >= 55:
        label = "Good"
    elif score >= 35:
        label = "Fair"
    else:
        label = "Poor"

    lang_instruction = (
        "Respond in simple Hindi (Devanagari script)."
        if req.language == "hindi"
        else "Respond in clear, simple English."
    )

    prompt = f"""You are a sleep health assistant. A patient's overnight EEG sleep study produced these results:

- Sleep Quality Score: {score:.1f}/100 ({label})
- Total Recording: {m.recording_minutes:.0f} minutes
- Total Sleep Time: {m.tst_min:.0f} minutes
- Sleep Efficiency: {m.sleep_efficiency*100:.1f}% (normal: >85%)
- Sleep Latency: {m.sleep_latency_min:.1f} minutes to fall asleep (normal: <20 min)
- WASO (wake after sleep onset): {m.waso_min:.0f} minutes (normal: <30 min)
- Deep Sleep (N3): {m.pct_n3*100:.1f}% of sleep (normal: 15-25%)
- REM Sleep: {m.pct_rem*100:.1f}% of sleep (normal: 20-25%)
- Fragmentation Index: {m.fragmentation_index*100:.1f}% (lower is better)
- Model Confidence: {m.confidence_mean*100:.0f}%

{lang_instruction}

Provide:
1. A 2-3 sentence plain-language explanation of what these results mean.
2. Exactly 3 specific, actionable tips to improve sleep quality.

Format as JSON only — no extra text:
{{"explanation": "...", "tips": ["tip1", "tip2", "tip3"]}}"""

    # Try Gemini first, then Anthropic, then rule-based fallback
    if state.gemini_model is not None:
        try:
            response = state.gemini_model.models.generate_content(
                model="gemini-3-flash-preview", contents=prompt
            )
            text  = response.text.strip()
            start = text.find("{")
            end   = text.rfind("}") + 1
            parsed = json.loads(text[start:end])
            return ExplainResponse(
                explanation  = parsed["explanation"],
                tips         = parsed["tips"],
                ai_available = True,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Gemini explain failed: %s", e, exc_info=True)

    if state.anthropic_client is not None:
        try:
            response = state.anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            text  = response.content[0].text.strip()
            start = text.find("{")
            end   = text.rfind("}") + 1
            parsed = json.loads(text[start:end])
            return ExplainResponse(
                explanation  = parsed["explanation"],
                tips         = parsed["tips"],
                ai_available = True,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Claude explain failed: %s", e, exc_info=True)

    explanation, tips = rule_based_explain(m, label)
    return ExplainResponse(explanation=explanation, tips=tips, ai_available=False)