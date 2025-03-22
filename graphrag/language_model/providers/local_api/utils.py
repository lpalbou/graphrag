# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Utils for the Local API provider."""

import asyncio
import threading
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")

_loop = asyncio.new_event_loop()
_thr = threading.Thread(target=_loop.run_forever, name="Local API Async Runner", daemon=True)


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