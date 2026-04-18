# ai_explainer.py
import asyncio
import logging
from config import GEMINI_API_KEY, MODEL_NAME, API_TIMEOUT

logger = logging.getLogger(__name__)

client = None
if GEMINI_API_KEY:
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
    except ImportError:
        logger.warning("google-generativeai package not installed")
    except Exception as e:
        logger.warning("Failed to initialize Gemini client: %s", e)


async def generate_ai_explanation(
    original_code: str,
    optimized_code: str,
    rules: list,
    speedup: float
) -> str:

    rules_text = "\n".join(
        [
            f"- {r.get('message', '')} -> {r.get('suggestion', '')}"
            for r in (rules or [])[:5]
        ]
    ) or "No specific rule patterns detected."

    speed_text = (
        f"{round(speedup, 2)}x speedup"
        if speedup and speedup > 1
        else "performance improvement"
    )

    if not client:
        return _generate_fallback(rules, speedup)

    from google.genai import types

    prompt = f"""You are a Python performance expert explaining code changes to a Python developer.

ORIGINAL CODE:
{original_code}

OPTIMIZED CODE:
{optimized_code}

PATTERNS DETECTED:
{rules_text}

MEASURED SPEEDUP:
{speed_text}

Respond in exactly this format with these three labeled sections:

**What changed:** One sentence describing the specific code transformation that was made.

**Why it's faster:** One sentence explaining why this change improves performance in Python terms only.

**Result:** One sentence stating the performance outcome and what this means for the code.

Strict rules:
- Never mention C, bytecode, interpreter internals, or low-level details
- Only use Python developer terminology
- Keep each section to exactly one sentence
- Do not add any extra text outside the three sections"""

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=2048
                    )
                )
            ),
            timeout=API_TIMEOUT
        )

        text = response.text.strip()

        if text.startswith("```"):
            text = text.replace("```python", "").replace("```", "").strip()

        return text if text else _generate_fallback(rules, speedup)

    except Exception as e:
        logger.warning("AI explanation generation failed: %s", e)
        return _generate_fallback(rules, speedup)


def _generate_fallback(rules: list, speedup: float) -> str:
    parts = []

    what = ""
    why = ""

    if rules:
        suggestions = [r.get("suggestion", "") for r in rules[:1] if r.get("suggestion")]
        detections = [r.get("message", "") for r in rules[:1] if r.get("message")]
        if suggestions:
            what = suggestions[0]
        if detections:
            why = detections[0]

    if not what:
        what = "The code was restructured to remove inefficient patterns."

    if not why:
        why = "This reduces the number of operations Python needs to perform per iteration."

    if speedup and speedup > 1:
        pct = round((speedup - 1) * 100, 1)
        result = f"The optimized version runs ~{round(speedup, 2)}x faster, a {pct}% improvement."
    else:
        result = "The optimized version is more efficient and easier to maintain."

    return f"**What changed:** {what}\n\n**Why it's faster:** {why}\n\n**Result:** {result}"
