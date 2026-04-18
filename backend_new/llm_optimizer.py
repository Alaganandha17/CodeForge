# llm_optimizer.py
import ast
import asyncio
import logging
from typing import List, Dict, Optional
from config import GEMINI_API_KEY, MODEL_NAME, API_TIMEOUT

logger = logging.getLogger(__name__)

# Only initialize client if API key is available
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


def _clean_markdown_fences(text: str) -> str:
    """Safely strip markdown code fences from LLM output."""
    text = text.strip()

    if text.startswith("```python"):
        text = text[len("```python"):]
        if "```" in text:
            text = text[:text.rfind("```")]
        return text.strip()

    if text.startswith("```"):
        text = text[3:]
        if "```" in text:
            text = text[:text.rfind("```")]
        return text.strip()

    return text


async def optimize_with_gemini(code: str, hints: Optional[List[Dict]] = None) -> str:
    if not client:
        raise Exception("Gemini API not configured (missing GEMINI_API_KEY)")

    from google.genai import types

    hint_text = ""
    if hints:
        hint_text = "\nDETECTED ISSUES:\n" + "\n".join(
            f"- {h.get('message', '')}" for h in hints
        ) + "\n"

    prompt = f"""You are an expert Python optimizer.
{hint_text}

CRITICAL RULES:
1. Keep ALL variable definitions and assignments intact (e.g., data = list(range(1000)))
2. Keep ALL import statements
3. Only optimize loops, comprehensions, and algorithms
4. Do NOT remove setup code or variable initializations

CODE:
{code}

Optimize this code for performance.
Return ONLY the optimized Python code. No explanations, no markdown fences."""

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2
                    )
                )
            ),
            timeout=API_TIMEOUT
        )

        optimized = _clean_markdown_fences(response.text)

        ast.parse(optimized)
        return optimized
    except asyncio.TimeoutError:
        logger.warning("Gemini API timeout (%ds)", API_TIMEOUT)
        raise Exception(f"Gemini API timeout ({API_TIMEOUT}s)")
    except SyntaxError:
        logger.warning("Gemini returned invalid Python, falling back to original")
        return code
    except Exception as e:
        logger.error("Gemini optimization failed: %s", e)
        raise Exception(f"Gemini optimization failed: {str(e)}")
