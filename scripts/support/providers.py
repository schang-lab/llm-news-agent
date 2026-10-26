"""Provider requests for cache misses. Keys are read only from the environment."""

import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .schemas import response_schema


def post(url, body, key_name):
    """POST JSON using an environment key, raising on transport or provider errors."""
    key = os.environ.get(key_name)
    if not key:
        raise ValueError(f"Cache miss: set {key_name} to make a provider request.")
    request = Request(
        url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    # No automatic retries; a failed request should not silently incur another charge.
    try:
        with urlopen(request, timeout=180) as response:
            result = json.loads(response.read())
    except HTTPError as error:
        raise RuntimeError(f"Provider HTTP {error.code}; no retry was made.") from None
    except (URLError, TimeoutError):
        raise RuntimeError("Provider connection failed; no retry was made.") from None
    if result.get("error"):
        raise RuntimeError("The provider returned an error; no retry was made.")
    return result


def call_model(req):
    """Send one uncached request and return raw text plus the provider response.

    Incomplete replies carry an error field so the cache can save them before raising.
    """
    redacted = r'(?:Article|Text):\n"""\n\[REDACTED\]\n"""$|articles you can read: \[REDACTED\]$'
    if any(re.search(redacted, message["content"].strip()) for message in req["messages"]):
        raise ValueError("Provide the original text, not [REDACTED], for a new model call.")
    model = req["model"]
    settings = dict(req["settings"])
    schema_name = settings.pop(
        "schema_name", "ArticleSelection" if req["stage"] == "retrieval" else "FrameExposureScores",
    )
    schema = response_schema(req)
    # Replicate identifies a local cache entry; it is not a provider parameter.
    messages = req["messages"]
    if model.startswith("gpt-"):
        body = dict(model=model, input=messages, store=False, **settings)
        if schema:
            body["text"] = {"format": dict(
                type="json_schema", name=schema_name, strict=True, schema=schema,
            )}
        response = post("https://api.openai.com/v1/responses", body, "OPENAI_API_KEY")
        text = "".join(
            part["text"] for item in response.get("output", [])
            for part in item.get("content", []) if part.get("type") == "output_text"
        )
        complete = response.get("status") == "completed"
    elif model == "claude-opus-4-6":
        body = dict(
            model="anthropic/claude-opus-4.6", max_tokens=4096,
            messages=[message for message in messages if message["role"] != "system"],
        )
        body.update(settings)
        body["system"] = [
            {"type": "text", "text": message["content"]}
            for message in messages if message["role"] == "system"
        ]
        if schema:
            body["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        response = post("https://openrouter.ai/api/v1/messages", body, "OPENROUTER_API_KEY")
        text = "".join(part["text"] for part in response.get("content", []) if part["type"] == "text")
        complete = response.get("stop_reason") in {"end_turn", "stop_sequence"}
    elif model == "gemini-3.1-pro-preview":
        body = dict(
            model="google/gemini-3.1-pro-preview", messages=messages,
            max_tokens=16384, stream=False, provider={"require_parameters": True},
        )
        body.update(settings)
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": dict(
                name=schema_name, strict=True, schema=schema,
            )}
        response = post("https://openrouter.ai/api/v1/chat/completions", body, "OPENROUTER_API_KEY")
        choice = response["choices"][0]
        text = choice["message"].get("content") or ""
        complete = choice.get("finish_reason") == "stop"
    else:
        raise ValueError(f"Unsupported model: {model}")
    result = dict(raw_output=text, provider_response=response)
    if not complete or not text.strip():
        result["error"] = "The provider returned an incomplete or empty response."
    return result


def embed(text):
    """Request one 3,072-dimensional embedding through OpenRouter."""
    response = post("https://openrouter.ai/api/v1/embeddings", dict(
        model="google/gemini-embedding-2", input=text, dimensions=3072,
        encoding_format="float", provider={"require_parameters": True},
    ), "OPENROUTER_API_KEY")
    return response["data"][0]["embedding"]
