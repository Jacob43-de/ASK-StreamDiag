"""
OpenAI-compatible clients for GPT-4 (judge) and GPT-3.5-turbo (dispatcher).
- API key from OPENAI_API_KEY env var
- base_url can be overridden by OPENAI_BASE_URL env var
- Retry with exponential backoff on transient errors
"""
import os
import time
from typing import Optional
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError

MAX_RETRIES = 5
BASE_BACKOFF = 2.0


def _make_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it before running ASK-StreamDiag pipeline."
        )
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    return OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)


def call_openai_chat(model: str, system_prompt: str, user_prompt: str,
                     temperature: float = 0.0,
                     max_retries: int = MAX_RETRIES) -> str:
    """
    Generic OpenAI chat completion call with retry.
    Returns stripped content or empty string on failure.
    """
    client = _make_client()
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            last_err = str(e)
            time.sleep(BASE_BACKOFF * (2 ** attempt))
            continue
        except Exception as e:
            last_err = str(e)
            time.sleep(BASE_BACKOFF * (2 ** attempt))
            continue
    print(f"[OpenAI:{model}] All retries failed. Last error: {last_err}")
    return ""


def gpt4_judge(prompt: str) -> str:
    """GPT-4 as strict judge, temperature=0.2, system prompt per paper §5.4."""
    return call_openai_chat(
        model="gpt-4",
        system_prompt="你是一个严格的评估专家。请只输出 JSON 格式。",
        user_prompt=prompt,
        temperature=0.2,
    )


def gpt35_dispatcher(prompt: str) -> str:
    """GPT-3.5-turbo as subtask decomposition dispatcher, temperature=0.0."""
    return call_openai_chat(
        model="gpt-3.5-turbo",
        system_prompt="你是一个任务分解专家。请将复杂任务拆解为原子子任务，只输出结构化的子任务列表。",
        user_prompt=prompt,
        temperature=0.0,
    )
