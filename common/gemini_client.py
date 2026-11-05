"""
Gemini 2.5 Flash HTTP client with retry logic.
- API key from environment variable GEMINI_API_KEY (never hardcoded)
- Endpoint: https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent
- Respects standard proxy env vars: HTTP_PROXY / HTTPS_PROXY / ALL_PROXY (requests default)
- Exponential backoff retry on 429 / 5xx / network errors
- Robust text extraction from candidates[0].content.parts
"""
import os
import time
import requests
from typing import Optional

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/gemini-2.5-flash:generateContent"
)
DEFAULT_TIMEOUT = 60  # seconds
MAX_RETRIES = 5
BASE_BACKOFF = 2.0  # seconds


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please set it before running experiments."
        )
    return key


def _extract_text(resp_json: dict) -> str:
    """Robustly extract text from Gemini response. Handles multi-part, safety blocks, empty."""
    try:
        candidates = resp_json.get("candidates", [])
        if not candidates:
            # Check promptFeedback for safety block
            return ""
        cand = candidates[0]
        # Finish reason could be SAFETY, RECITATION, etc.
        finish_reason = cand.get("finishReason", "")
        content = cand.get("content", {})
        parts = content.get("parts", [])
        texts = []
        for p in parts:
            # Skip Gemini 2.5 thinking blocks (marked with thought: true)
            if p.get("thought") is True:
                continue
            t = p.get("text", "")
            if t:
                texts.append(t)
        return "\n".join(texts).strip()
    except Exception:
        return ""


def call_gemini(prompt: str, temperature: float = 0.0,
                timeout: int = DEFAULT_TIMEOUT,
                max_retries: int = MAX_RETRIES) -> str:
    """
    Call Gemini 2.5 Flash with a single prompt string.
    Returns stripped text, or empty string on failure.
    """
    api_key = _get_api_key()
    url = f"{GEMINI_ENDPOINT}?key={api_key}"
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            # Retry on 429 / 5xx
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                last_err = f"HTTP {resp.status_code}"
                time.sleep(BASE_BACKOFF * (2 ** attempt))
                continue
            resp.raise_for_status()
            data = resp.json()
            return _extract_text(data)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            last_err = str(e)
            time.sleep(BASE_BACKOFF * (2 ** attempt))
            continue
        except Exception as e:
            last_err = str(e)
            time.sleep(BASE_BACKOFF * (2 ** attempt))
            continue

    # All retries exhausted
    print(f"[Gemini] All retries failed. Last error: {last_err}")
    return ""


class GeminiClient:
    """Stateful wrapper so method scripts can pass around a single callable."""

    def __init__(self, temperature: float = 0.0):
        self.temperature = temperature
        self.call_count = 0
        self.retry_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        return call_gemini(prompt, temperature=self.temperature)
