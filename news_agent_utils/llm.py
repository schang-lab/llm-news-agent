"""Shared LLM cache, client, and provider-call helpers.

The notebooks use three LLM interaction modes:
1. structured OpenAI parsing for framing annotation;
2. provider tool/function calls for retrieval article selection;
3. provider plain-text generation for synthesis.

The helpers below keep those modes explicit while sharing cache-key construction,
cache read/write behavior, API-key/client loading, and provider response parsing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .io import read_json, stable_hash, to_serializable, utc_now_iso, write_json

PROVIDER_API_KEY_NAMES = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def load_provider_api_key(provider: str) -> str | None:
    """Load the single API key for a provider from the environment."""
    key_name = PROVIDER_API_KEY_NAMES.get(provider)
    if key_name is None:
        raise ValueError(f"Unsupported model provider: {provider}")
    value = os.environ.get(key_name)
    return value.strip() if value and value.strip() else None


def create_provider_client(
    provider: str,
    api_key: str | None,
    *,
    openai_client_cls: Any = None,
    anthropic_module: Any = None,
    genai_module: Any = None,
) -> Any | None:
    """Create one provider client from one API key; return None when unavailable."""
    if not api_key:
        return None
    if provider == "openai":
        if openai_client_cls is None:
            return None
        return openai_client_cls(api_key=api_key)
    if provider == "anthropic":
        if anthropic_module is None:
            return None
        return anthropic_module.Anthropic(api_key=api_key)
    if provider == "gemini":
        if genai_module is None:
            return None
        return genai_module.Client(api_key=api_key)
    raise ValueError(f"Unsupported model provider: {provider}")


def get_provider_client(
    model_provider: str,
    provider_clients: Mapping[str, Any | None],
) -> Any | None:
    """Return the single configured client for a provider."""
    if model_provider not in provider_clients:
        raise ValueError(f"Unsupported model provider: {model_provider}")
    return provider_clients[model_provider]


def build_llm_cache_key(
    *,
    prompt: str,
    model_name: str,
    config_payload: Dict[str, Any],
    output_mode: str,
    model_provider: str | None = None,
) -> str:
    """Build a stable cache key using the input prompt to the model, the model name, and any associated configuration configuration."""
    payload = {"prompt": prompt}
    if model_provider is not None:
        payload["model_provider"] = model_provider
    payload.update(
        {
            "model": model_name,
            "config": to_serializable(config_payload),
            "output_mode": output_mode,
        }
    )
    return stable_hash(payload)


def dump_provider_response(response: Any) -> Any:
    """Convert provider SDK response objects into JSON-serializable values."""
    if hasattr(response, "model_dump"):
        return to_serializable(response.model_dump(mode="json"))
    if hasattr(response, "to_dict"):
        return to_serializable(response.to_dict())
    if hasattr(response, "to_json_dict"):
        return to_serializable(response.to_json_dict())
    return to_serializable(response)


def split_chat_input(input_payload: List[Dict[str, Any]]) -> tuple[str, str]:
    """Split a chat-style input payload into system text and user text."""
    system_prompt = ""
    user_prompt_parts = []
    for item in input_payload:
        role = item.get("role")
        content = item.get("content", "")
        if role == "system":
            system_prompt = str(content)
        elif role == "user":
            user_prompt_parts.append(str(content))
    return system_prompt, "\n\n".join(user_prompt_parts)


def read_existing_llm_cache(
    spec: Dict[str, Any],
    *,
    cache_read_enabled: bool = True,
) -> Dict[str, Any] | None:
    """Read a completed or pending cache row, returning None for misses/invalid JSON."""
    cache_path = Path(spec["cache_path"])
    if not cache_read_enabled or not cache_path.exists():
        return None
    try:
        return read_json(cache_path)
    except Exception as exc:
        print(f"Ignoring invalid LLM cache file and refreshing it: {cache_path} ({exc})")
        return None


def cache_has_field(cache_row: Dict[str, Any] | None, field_name: str) -> bool:
    """Return True when a cache row contains the completion field for a mode."""
    return isinstance(cache_row, dict) and field_name in cache_row


# ---------------------------------------------------------------------------
# Mode 1: OpenAI structured parsing for framing annotation.
# ---------------------------------------------------------------------------

def dump_parsed_response(parsed: Any) -> Dict[str, Any]:
    """Normalize OpenAI parsed structured output."""
    if hasattr(parsed, "model_dump"):
        return parsed.model_dump()
    if isinstance(parsed, dict):
        return parsed
    return to_serializable(parsed)


def run_cached_openai_structured_call(
    *,
    prompt: str,
    model_name: str,
    temperature: float | None,
    schema_name: str,
    response_schema: Any,
    llm_cache_dir: Path,
    openai_client: Any,
    cache_read_enabled: bool = True,
) -> tuple[Dict[str, Any], bool]:
    """Run an OpenAI structured parse call with cache lookup/write-through behavior."""
    config_payload = {
        "temperature": temperature,
        "schema_name": schema_name,
        "api": "responses.parse",
    }
    normalized_config = to_serializable(config_payload)
    output_mode = f"structured:{schema_name}"
    cache_key = build_llm_cache_key(
        prompt=prompt,
        model_name=model_name,
        config_payload=normalized_config,
        output_mode=output_mode,
    )
    cache_path = Path(llm_cache_dir) / f"{cache_key}.json"
    spec = {"cache_path": cache_path}
    cache_row = read_existing_llm_cache(spec, cache_read_enabled=cache_read_enabled)
    if cache_has_field(cache_row, "response"):
        return cache_row, True

    request_kwargs = {
        "model": model_name,
        "input": [{"role": "user", "content": prompt}],
        "text_format": response_schema,
    }
    if temperature is not None:
        request_kwargs["temperature"] = temperature

    response = openai_client.responses.parse(**request_kwargs)
    payload = dump_parsed_response(response.output_parsed)
    row = {
        "cache_key": cache_key,
        "model": model_name,
        "config": normalized_config,
        "output_mode": output_mode,
        "prompt": prompt,
        "generated_at": utc_now_iso(),
        "response": to_serializable(payload),
        "response_id": getattr(response, "id", None),
    }
    write_json(cache_path, row)
    return row, False


# ---------------------------------------------------------------------------
# Mode 2: provider tool/function calls for retrieval article selection.
# ---------------------------------------------------------------------------

def dump_response_output(response: Any) -> List[Dict[str, Any]]:
    """Extract OpenAI Responses API output items as serializable dictionaries."""
    output = []
    for item in getattr(response, "output", []) or []:
        if hasattr(item, "model_dump"):
            output.append(to_serializable(item.model_dump(mode="json")))
        elif isinstance(item, dict):
            output.append(item)
        else:
            output.append(to_serializable(item))
    return output


def normalize_openai_function_calls(response: Any) -> List[Dict[str, Any]]:
    """Normalize OpenAI function-call output into a shared format."""
    return dump_response_output(response)


def normalize_anthropic_function_calls(response: Any) -> List[Dict[str, Any]]:
    """Normalize Anthropic tool-use blocks into a shared format."""
    normalized = []
    for item in getattr(response, "content", []) or []:
        if getattr(item, "type", None) != "tool_use":
            continue
        normalized.append(
            {
                "type": "function_call",
                "name": getattr(item, "name", ""),
                "call_id": getattr(item, "id", None),
                "arguments": getattr(item, "input", {}) or {},
            }
        )
    return normalized


def normalize_gemini_function_calls(response: Any) -> List[Dict[str, Any]]:
    """Normalize Gemini function-call objects into a shared format."""
    normalized = []
    for function_call in getattr(response, "function_calls", []) or []:
        args = getattr(function_call, "args", {}) or {}
        normalized.append(
            {
                "type": "function_call",
                "name": getattr(function_call, "name", ""),
                "call_id": getattr(function_call, "id", None),
                "arguments": dict(args) if hasattr(args, "items") else args,
            }
        )
    return normalized


def make_anthropic_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an OpenAI-style tool schema into Anthropic's tool schema."""
    return {"name": tool["name"], "description": tool["description"], "input_schema": tool["parameters"]}


def make_gemini_tool(tool: Dict[str, Any], genai_types: Any) -> Any:
    """Convert an OpenAI-style tool schema into a Gemini FunctionDeclaration tool."""
    return genai_types.Tool(
        function_declarations=[
            genai_types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters_json_schema=tool["parameters"],
            )
        ]
    )


def build_openai_tool_request_body(
    *,
    model_name: str,
    input_payload: List[Dict[str, Any]],
    temperature: float | None,
    tools: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the OpenAI Responses API request body for retrieval tool selection."""
    request_body = {"model": model_name, "input": input_payload}
    if tools:
        request_body["tools"] = tools
    if temperature is not None:
        request_body["temperature"] = temperature
    return request_body


def build_anthropic_tool_request_body(
    *,
    model_name: str,
    input_payload: List[Dict[str, Any]],
    temperature: float | None,
    tools: List[Dict[str, Any]],
    tool_name: str,
) -> Dict[str, Any]:
    """Build the Anthropic Messages API request body for retrieval tool selection."""
    system_prompt, user_prompt = split_chat_input(input_payload)
    request_body = {
        "model": model_name,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "tools": [make_anthropic_tool(tool) for tool in tools],
        "tool_choice": {"type": "tool", "name": tool_name},
    }
    if temperature is not None:
        request_body["temperature"] = temperature
    return request_body


def _build_tool_selection_request_spec(
    *,
    prompt: str,
    input_payload: List[Dict[str, Any]],
    model_provider: str,
    model_name: str,
    temperature: float | None,
    tools: List[Dict[str, Any]] | None,
    output_mode: str,
    tool_schema_version: str,
    tool_name: str,
    llm_cache_dir: Path,
) -> Dict[str, Any]:
    """Build cache metadata and normalized config for a tool-selection request."""
    provider_api_names = {
        "openai": "responses.create",
        "anthropic": "messages.create",
        "gemini": "models.generate_content",
    }
    if model_provider not in provider_api_names:
        raise ValueError(f"Unsupported model provider: {model_provider}")

    normalized_tools = tools or []
    config_payload = {
        "temperature": temperature,
        "api": provider_api_names[model_provider],
        "tool_schema_version": tool_schema_version,
        "tools": normalized_tools,
        "tool_config": {
            "openai": {"tool_choice": "auto"},
            "anthropic": {"tool_choice": {"type": "tool", "name": tool_name}},
            "gemini": {
                "function_calling_config": {"mode": "ANY", "allowed_function_names": [tool_name]},
                "automatic_function_calling": {"disable": True},
            },
        }[model_provider],
    }
    normalized_config = to_serializable(config_payload)
    cache_key = build_llm_cache_key(
        prompt=prompt,
        model_provider=model_provider,
        model_name=model_name,
        config_payload=normalized_config,
        output_mode=output_mode,
    )
    return {
        "cache_key": cache_key,
        "cache_path": Path(llm_cache_dir) / f"{cache_key}.json",
        "model_provider": model_provider,
        "model": model_name,
        "config": normalized_config,
        "output_mode": output_mode,
        "prompt": prompt,
        "input_payload": input_payload,
        "temperature": temperature,
        "tools": normalized_tools,
        "tool_name": tool_name,
    }


def run_provider_tool_selection_call(
    *,
    client: Any,
    model_provider: str,
    model_name: str,
    input_payload: List[Dict[str, Any]],
    temperature: float | None,
    tools: List[Dict[str, Any]],
    tool_name: str,
    genai_types: Any = None,
) -> tuple[Any, List[Dict[str, Any]], Any]:
    """Call one provider for retrieval tool selection and normalize its tool calls."""
    if model_provider == "openai":
        response = client.responses.create(
            **build_openai_tool_request_body(
                model_name=model_name,
                input_payload=input_payload,
                temperature=temperature,
                tools=tools,
            )
        )
        return response, normalize_openai_function_calls(response), getattr(response, "id", None)
    if model_provider == "anthropic":
        response = client.messages.create(
            **build_anthropic_tool_request_body(
                model_name=model_name,
                input_payload=input_payload,
                temperature=temperature,
                tools=tools,
                tool_name=tool_name,
            )
        )
        return response, normalize_anthropic_function_calls(response), getattr(response, "id", None)
    if model_provider == "gemini":
        if genai_types is None:
            raise ImportError("google-genai is required for Gemini tool selection calls.")
        system_prompt, user_prompt = split_chat_input(input_payload)
        config_kwargs = {
            "system_instruction": system_prompt,
            "tools": [make_gemini_tool(tool, genai_types) for tool in tools],
            "tool_config": genai_types.ToolConfig(
                function_calling_config=genai_types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=[tool_name],
                )
            ),
            "automatic_function_calling": genai_types.AutomaticFunctionCallingConfig(disable=True),
        }
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
        response_id = getattr(response, "response_id", None) or getattr(response, "id", None)
        return response, normalize_gemini_function_calls(response), response_id
    raise ValueError(f"Unsupported model provider: {model_provider}")


def run_cached_tool_selection_call(
    *,
    prompt: str,
    input_payload: List[Dict[str, Any]],
    model_provider: str,
    model_name: str,
    temperature: float | None,
    tools: List[Dict[str, Any]] | None,
    output_mode: str,
    tool_schema_version: str,
    tool_name: str,
    llm_cache_dir: Path,
    provider_clients: Mapping[str, Any | None],
    genai_types: Any = None,
    cache_read_enabled: bool = True,
) -> tuple[Dict[str, Any], bool]:
    """Run a tool-selection call with cache lookup/write-through behavior."""
    spec = _build_tool_selection_request_spec(
        prompt=prompt,
        input_payload=input_payload,
        model_provider=model_provider,
        model_name=model_name,
        temperature=temperature,
        tools=tools,
        output_mode=output_mode,
        tool_schema_version=tool_schema_version,
        tool_name=tool_name,
        llm_cache_dir=llm_cache_dir,
    )
    cache_row = read_existing_llm_cache(spec, cache_read_enabled=cache_read_enabled)
    if cache_has_field(cache_row, "response"):
        return cache_row, True

    client = get_provider_client(model_provider, provider_clients)
    if client is None:
        raise ValueError(
            f"No API key was found for provider {model_provider!r}. "
            f"Set {PROVIDER_API_KEY_NAMES[model_provider]} before running this model."
        )
    response, normalized_response, response_id = run_provider_tool_selection_call(
        client=client,
        model_provider=model_provider,
        model_name=model_name,
        input_payload=input_payload,
        temperature=temperature,
        tools=spec["tools"],
        tool_name=tool_name,
        genai_types=genai_types,
    )
    row = {
        "cache_key": spec["cache_key"],
        "model_provider": model_provider,
        "model": model_name,
        "config": spec["config"],
        "output_mode": output_mode,
        "prompt": prompt,
        "input_payload": to_serializable(input_payload),
        "generated_at": utc_now_iso(),
        "response": normalized_response,
        "raw_response": dump_provider_response(response),
        "response_id": response_id,
    }
    write_json(Path(spec["cache_path"]), row)
    return row, False


# ---------------------------------------------------------------------------
# Mode 3: provider text generation for synthesis.
# ---------------------------------------------------------------------------

def extract_openai_response_text(response: Any, response_dump: Dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI Responses API response."""
    output_text = getattr(response, "output_text", None) or response_dump.get("output_text")
    if output_text:
        return str(output_text)
    text_parts = []
    for output_item in response_dump.get("output", []) or []:
        for content_item in output_item.get("content", []) or []:
            if content_item.get("type") in {"output_text", "text"}:
                text = content_item.get("text")
                if text:
                    text_parts.append(str(text))
    return "\n".join(text_parts).strip()


def extract_anthropic_response_text(response: Any) -> str:
    """Extract assistant text from an Anthropic Messages API response."""
    text_parts = []
    for item in getattr(response, "content", []) or []:
        if getattr(item, "type", None) == "text":
            text_parts.append(str(getattr(item, "text", "")))
    return "\n".join(part for part in text_parts if part).strip()


def extract_gemini_response_text(response: Any) -> str:
    """Extract assistant text from a Gemini generate_content response."""
    response_text = getattr(response, "text", None)
    if response_text:
        return str(response_text)
    text_parts = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                text_parts.append(str(text))
    return "\n".join(text_parts).strip()


def _build_text_generation_request_spec(
    *,
    prompt: str,
    input_payload: List[Dict[str, Any]],
    model_provider: str,
    model_name: str,
    temperature: float | None,
    output_mode: str,
    max_tokens: int,
    llm_cache_dir: Path,
) -> Dict[str, Any]:
    """Build cache metadata and normalized config for a text-generation request."""
    provider_api_names = {
        "openai": "responses.create",
        "anthropic": "messages.create",
        "gemini": "models.generate_content",
    }
    if model_provider not in provider_api_names:
        raise ValueError(f"Unsupported model provider: {model_provider}")
    config_payload = {
        "temperature": temperature,
        "api": provider_api_names[model_provider],
        "max_tokens": max_tokens if model_provider == "anthropic" else None,
    }
    normalized_config = to_serializable(config_payload)
    cache_key = build_llm_cache_key(
        prompt=prompt,
        model_provider=model_provider,
        model_name=model_name,
        config_payload=normalized_config,
        output_mode=output_mode,
    )
    return {
        "cache_key": cache_key,
        "cache_path": Path(llm_cache_dir) / f"{cache_key}.json",
        "model_provider": model_provider,
        "model": model_name,
        "config": normalized_config,
        "output_mode": output_mode,
        "prompt": prompt,
        "input_payload": input_payload,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def run_provider_text_generation_call(
    *,
    client: Any,
    model_provider: str,
    model_name: str,
    input_payload: List[Dict[str, Any]],
    temperature: float | None,
    max_tokens: int,
    genai_types: Any = None,
) -> tuple[Any, str, Any]:
    """Call one provider for text generation and return raw response, text, and id."""
    if model_provider == "openai":
        request_body = {"model": model_name, "input": input_payload}
        if temperature is not None:
            request_body["temperature"] = temperature
        response = client.responses.create(**request_body)
        response_dump = dump_provider_response(response)
        return response, extract_openai_response_text(response, response_dump), getattr(response, "id", None)
    if model_provider == "anthropic":
        system_prompt, user_prompt = split_chat_input(input_payload)
        request_body = {
            "model": model_name,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if temperature is not None:
            request_body["temperature"] = temperature
        response = client.messages.create(**request_body)
        return response, extract_anthropic_response_text(response), getattr(response, "id", None)
    if model_provider == "gemini":
        if genai_types is None:
            raise ImportError("google-genai is required for Gemini generation calls.")
        system_prompt, user_prompt = split_chat_input(input_payload)
        config_kwargs = {"system_instruction": system_prompt}
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
        response_id = getattr(response, "response_id", None) or getattr(response, "id", None)
        return response, extract_gemini_response_text(response), response_id
    raise ValueError(f"Unsupported model provider: {model_provider}")


def run_cached_text_generation_call(
    *,
    prompt: str,
    input_payload: List[Dict[str, Any]],
    model_provider: str,
    model_name: str,
    temperature: float | None,
    output_mode: str,
    max_tokens: int,
    llm_cache_dir: Path,
    provider_clients: Mapping[str, Any | None],
    genai_types: Any = None,
    cache_read_enabled: bool = True,
) -> tuple[Dict[str, Any], bool]:
    """Run a text-generation call with cache lookup/write-through behavior."""
    spec = _build_text_generation_request_spec(
        prompt=prompt,
        input_payload=input_payload,
        model_provider=model_provider,
        model_name=model_name,
        temperature=temperature,
        output_mode=output_mode,
        max_tokens=max_tokens,
        llm_cache_dir=llm_cache_dir,
    )
    cache_row = read_existing_llm_cache(spec, cache_read_enabled=cache_read_enabled)
    if cache_has_field(cache_row, "response_text"):
        return cache_row, True

    client = get_provider_client(model_provider, provider_clients)
    if client is None:
        raise ValueError(
            f"No API key was found for provider {model_provider!r}. "
            f"Set {PROVIDER_API_KEY_NAMES[model_provider]} before running this model."
        )
    response, response_text, response_id = run_provider_text_generation_call(
        client=client,
        model_provider=model_provider,
        model_name=model_name,
        input_payload=input_payload,
        temperature=temperature,
        max_tokens=max_tokens,
        genai_types=genai_types,
    )
    row = {
        "cache_key": spec["cache_key"],
        "model_provider": model_provider,
        "model": model_name,
        "config": spec["config"],
        "output_mode": output_mode,
        "prompt": prompt,
        "input_payload": to_serializable(input_payload),
        "generated_at": utc_now_iso(),
        "response_text": response_text,
        "raw_response": dump_provider_response(response),
        "response_id": response_id,
    }
    write_json(Path(spec["cache_path"]), row)
    return row, False
