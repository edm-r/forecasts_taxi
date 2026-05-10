"""
src/core/decorators.py
----------------------
Décorateurs avancés pour instrumenter le pipeline NYC Taxi.

Exercice 1.7 — Master 1 IABD
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import time
from collections import OrderedDict
from typing import Any, Callable

import pandas as pd


_UNIT_FACTORS = {
    "s": 1.0,
    "ms": 1000.0,
    "us": 1_000_000.0,
}


def timeit(
    func: Callable[..., Any] | None = None,
    *,
    unit: str = "ms",
) -> Callable[..., Any]:
    """
    Mesure le temps d'exécution d'une fonction.

    Supporte ``@timeit`` et ``@timeit(unit="ms")``.
    """

    if unit not in _UNIT_FACTORS:
        raise ValueError(f"Unité non supportée: {unit!r}. Choix: {sorted(_UNIT_FACTORS)}.")

    def decorator(inner_func: Callable[..., Any]) -> Callable[..., Any]:
        logger = logging.getLogger(inner_func.__module__)

        @functools.wraps(inner_func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started_at = time.perf_counter()
            result = inner_func(*args, **kwargs)
            elapsed_s = time.perf_counter() - started_at
            elapsed = elapsed_s * _UNIT_FACTORS[unit]
            logger.info("[timeit] %s took %.3f %s", inner_func.__name__, elapsed, unit)
            return result

        return wrapper

    if func is not None:
        return decorator(func)

    return decorator


def log_calls(
    *,
    level: str = "INFO",
    log_args: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Logue les appels de fonction via ``logging``.
    """

    log_level = getattr(logging, level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Niveau de log invalide: {level!r}.")

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        logger = logging.getLogger(func.__module__)
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if log_args:
                bound = signature.bind_partial(*args, **kwargs)
                payload = ", ".join(f"{name}={value!r}" for name, value in bound.arguments.items())
                message = f"[log_calls] {func.__name__}({payload})"
            else:
                message = f"[log_calls] {func.__name__}()"

            logger.log(log_level, message)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def retry(
    *,
    max_attempts: int = 3,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (IOError,),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Réessaie une fonction en cas d'exception, avec backoff exponentiel.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts doit être >= 1.")
    if backoff <= 0:
        raise ValueError("backoff doit être > 0.")

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        logger = logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            while True:
                attempt += 1
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt >= max_attempts:
                        raise

                    delay = backoff ** (attempt - 1)
                    logger.warning(
                        "[retry] %s failed on attempt %d/%d with %r. Retry in %.3f s.",
                        func.__name__,
                        attempt,
                        max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


def memoize_dataframe(
    *,
    maxsize: int = 4,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Cache LRU spécialisé pour les fonctions recevant des DataFrames.
    """

    if maxsize < 1:
        raise ValueError("maxsize doit être >= 1.")

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        cache: OrderedDict[Any, Any] = OrderedDict()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (_freeze_for_cache(args), _freeze_for_cache(kwargs))
            if key in cache:
                cache.move_to_end(key)
                return _copy_if_pandas(cache[key])

            result = func(*args, **kwargs)
            cache[key] = _copy_if_pandas(result)
            cache.move_to_end(key)
            if len(cache) > maxsize:
                cache.popitem(last=False)
            return _copy_if_pandas(result)

        return wrapper

    return decorator


def _freeze_for_cache(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return ("dataframe", _hash_dataframe(value))
    if isinstance(value, pd.Series):
        return ("series", _hash_series(value))
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_for_cache(val)) for key, val in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_cache(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_for_cache(item) for item in value))
    return value


def _hash_dataframe(df: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    digest.update(repr(tuple(df.columns)).encode("utf-8"))
    digest.update(repr(tuple(map(str, df.dtypes))).encode("utf-8"))
    return digest.hexdigest()


def _hash_series(series: pd.Series) -> str:
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(series, index=True).values.tobytes())
    digest.update(repr(series.name).encode("utf-8"))
    digest.update(str(series.dtype).encode("utf-8"))
    return digest.hexdigest()


def _copy_if_pandas(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    if isinstance(value, pd.Series):
        return value.copy(deep=True)
    return value
