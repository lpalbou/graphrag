# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Utils for the Anthropic provider."""

import asyncio
import threading
from typing import Any, Coroutine, Dict, List, Optional, TypeVar, cast

T = TypeVar("T")

_loop = asyncio.new_event_loop()
_thr = threading.Thread(target=_loop.run_forever, name="Anthropic Async Runner", daemon=True)


def run_coroutine_sync(coroutine: Coroutine[Any, Any, T]) -> T:
    """
    Run a coroutine synchronously.

    Args:
        coroutine: The coroutine to run.

    Returns
    -------
        The result of the coroutine.
    """
    if not _thr.is_alive():
        _thr.start()
    future = asyncio.run_coroutine_threadsafe(coroutine, _loop)
    return future.result()


def convert_history_to_anthropic_format(
    prompt: str, history: Optional[List] = None
) -> List[Dict[str, Any]]:
    """
    Convert prompt and history to Anthropic message format.

    Args:
        prompt: The current prompt
        history: The message history if any

    Returns:
        A list of messages in Anthropic format
    """
    messages = []
    
    # Add history if provided
    if history:
        for item in history:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                
                if role.lower() in ["user", "assistant", "system"]:
                    messages.append({
                        "role": role.lower(),
                        "content": content
                    })
            elif isinstance(item, list) and len(item) >= 2:
                # Handle list format [role, content]
                role, content = item[0], item[1]
                
                if role.lower() in ["user", "assistant", "system"]:
                    messages.append({
                        "role": role.lower(),
                        "content": content
                    })
    
    # Add current prompt as user message if not empty
    if prompt:
        messages.append({
            "role": "user",
            "content": prompt
        })
    
    # Ensure we have at least one message
    if not messages:
        messages.append({
            "role": "user",
            "content": prompt or ""
        })
    
    return messages 