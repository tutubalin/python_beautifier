"""Keep deeply nested input from crashing the interpreter.

Generated code sometimes contains very long ``elif`` chains or ``a + b + c + ...``
expressions.  Their syntax trees are thousands of levels deep, which overflows
Python's default recursion limit.  :func:`run_deep` raises the limit and, when
the platform allows it, runs the work on a thread with a large stack so that a
deep recursion ends in a normal ``RecursionError`` instead of a crash.
"""
from __future__ import annotations

import sys
import threading
from typing import Any, Callable

DEEP_RECURSION_LIMIT = 20_000
DEEP_STACK_BYTES = 256 * 1024 * 1024


def run_deep(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn(*args, **kwargs)`` with a generous recursion limit and stack."""
    box: dict = {}

    def target() -> None:
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # re-raised on the calling thread
            box["error"] = exc

    old_limit = sys.getrecursionlimit()
    thread = None
    try:
        sys.setrecursionlimit(max(old_limit, DEEP_RECURSION_LIMIT))
        previous = None
        try:
            previous = threading.stack_size(DEEP_STACK_BYTES)
            thread = threading.Thread(target=target, name="pybeautify", daemon=True)
            thread.start()
        except (ValueError, RuntimeError, MemoryError):
            thread = None  # no big stack available here (e.g. WebAssembly): stay on this thread
        finally:
            if previous is not None:
                threading.stack_size(previous)
        if thread is None:
            sys.setrecursionlimit(old_limit)  # no big stack: never ask for more depth than we have room for
            target()
        else:
            thread.join()
    finally:
        sys.setrecursionlimit(old_limit)
    if "error" in box:
        raise box["error"]
    return box["value"]
