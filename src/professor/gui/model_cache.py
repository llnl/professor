"""Caches loaded professor models and model evaluation results for the GUI."""

from __future__ import annotations

import base64
import gc
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from professor.vela import config, model

DEFAULT_CACHE_ENTRIES = 2
DEFAULT_RESULT_CACHE_ENTRIES = 8


@dataclass
class CachedModel:
    """A loaded model and its session usage metadata."""

    key: str
    helper: ProfessorHelper
    ref_count: int = 0
    last_access: float = 0.0


@dataclass
class CachedResult:
    """A model result's metadata and cached output arrays."""

    token: str
    instance_id: str
    fields: list[str]
    shape: list[int]
    arrays: dict[str, np.ndarray]
    last_access: float = 0.0


class ProfessorHelper:
    """
    Helper class for launching and evaluating one professor model.
    """

    def __init__(self, prof_config: config.Config) -> None:
        """Create a helper for the supplied professor configuration."""
        self._config = prof_config
        self._device: torch.device | str = ""
        self._model: torch.nn.Module | None = None

    def close(self) -> None:
        """Release the loaded model and clear unused CUDA memory."""
        self._model = None
        self._config = None  # type: ignore[assignment]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, model_args: Iterable[float]) -> np.ndarray:
        """
        Evaluate the loaded professor model.
        """
        if self._config is None:
            return np.zeros(0)

        if self._model is None:
            loaded_model = model.PyTorchModel(self._config)
            self._device = loaded_model._get_checkpoint_device("0")
            self._model = loaded_model.models[0].ref

        with torch.no_grad():
            tmp = np.array(model_args)
            x = torch.Tensor(np.reshape(tmp, (1, -1, 1, 1)))
            if self._config.half_precision:
                x = x.half()
            y = self._model(x.to(self._device)).detach().to("cpu").numpy().astype(np.float32)
            return y


class ModelCache:
    """Manage loaded models with session references and bounded LRU eviction."""

    def __init__(self, max_entries: int = DEFAULT_CACHE_ENTRIES) -> None:
        """Initialize the cache with a maximum number of model entries."""
        self._max_entries = max_entries
        self._entries: OrderedDict[str, CachedModel] = OrderedDict()
        self._session_keys: dict[str, str] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    def release_session(self, session_id: str) -> None:
        """Drop a session's model reference and evict unused entries if needed."""
        with self._lock:
            old_key = self._session_keys.pop(session_id, None)
            if old_key is None:
                return
            entry = self._entries.get(old_key)
            if entry is not None:
                entry.ref_count = max(0, entry.ref_count - 1)
        self._evict_if_needed()

    def get_or_load(self, payload: dict[str, Any], session_id: str) -> ProfessorHelper:
        """Return the cached model for a session, loading it when absent.

        The payload must include ``cache_key``, ``yaml_text``, and ``source``.
        A session is associated with at most one model cache key at a time.
        """
        cache_key = payload["cache_key"]
        key_lock = self._get_key_lock(cache_key)
        with key_lock:
            entry = self._get_entry(cache_key)
            if entry is None:
                entry = self._load_entry(payload)
                with self._lock:
                    self._entries[cache_key] = entry

            with self._lock:
                previous_key = self._session_keys.get(session_id)
                if previous_key != cache_key:
                    self._release_previous_locked(previous_key)
                    self._session_keys[session_id] = cache_key
                    entry.ref_count += 1
                entry.last_access = time.monotonic()
                self._entries.move_to_end(cache_key)

        self._evict_if_needed()
        return entry.helper

    def _get_entry(self, cache_key: str) -> CachedModel | None:
        """Look up an entry and mark it as the most recently accessed."""
        with self._lock:
            entry = self._entries.get(cache_key)
            if entry is not None:
                entry.last_access = time.monotonic()
                self._entries.move_to_end(cache_key)
            return entry

    def _get_key_lock(self, cache_key: str) -> threading.Lock:
        """Get or create the lock used to serialize loads for one cache key."""
        with self._lock:
            lock = self._locks.get(cache_key)
            if lock is None:
                lock = threading.Lock()
                self._locks[cache_key] = lock
            return lock

    def _load_entry(self, payload: dict[str, Any]) -> CachedModel:
        """Construct a cache entry from a model configuration payload."""
        prof_config = config.Config.from_text(payload["yaml_text"], source=payload["source"])
        helper = ProfessorHelper(prof_config)
        return CachedModel(key=payload["cache_key"], helper=helper, last_access=time.monotonic())

    def _release_previous_locked(self, previous_key: str | None) -> None:
        """Decrement a previous model's reference count while holding the lock."""
        if previous_key is None:
            return
        previous_entry = self._entries.get(previous_key)
        if previous_entry is None:
            return
        previous_entry.ref_count = max(0, previous_entry.ref_count - 1)

    def _evict_if_needed(self) -> None:
        """Close least-recently-used unreferenced entries until within capacity."""
        evicted: list[CachedModel] = []
        with self._lock:
            while len(self._entries) > self._max_entries:
                evict_key = None
                for key, entry in self._entries.items():
                    if entry.ref_count == 0:
                        evict_key = key
                        break
                if evict_key is None:
                    return
                evicted.append(self._entries.pop(evict_key))
                self._locks.pop(evict_key, None)

        for entry in evicted:
            entry.helper.close()


model_cache = ModelCache()


class ResultCache:
    """Keep recent model outputs addressable by token and instance identifier."""

    def __init__(self, max_entries: int = DEFAULT_RESULT_CACHE_ENTRIES) -> None:
        """Initialize the result cache with a maximum number of results."""
        self._max_entries = max_entries
        self._results: OrderedDict[str, CachedResult] = OrderedDict()
        self._instance_tokens: dict[str, str] = {}
        self._lock = threading.Lock()

    def put(self, instance_id: str, fields: list[str], output: np.ndarray) -> dict[str, Any]:
        """Store output arrays and return their token, shape, and field names.

        Each field's array is copied from the corresponding channel in the
        batch's first item. A newer result replaces any result for the instance.
        """
        token = str(uuid.uuid4())
        shape = list(np.shape(output)[2:])
        arrays = {}
        for index, field in enumerate(fields):
            arrays[field] = np.array(output[0, index, ...], dtype=np.float32, copy=True)

        result = CachedResult(
            token=token,
            instance_id=instance_id,
            fields=fields,
            shape=shape,
            arrays=arrays,
            last_access=time.monotonic(),
        )
        evicted = []
        with self._lock:
            previous_token = self._instance_tokens.get(instance_id)
            if previous_token is not None:
                self._results.pop(previous_token, None)
            self._instance_tokens[instance_id] = token
            self._results[token] = result
            while len(self._results) > self._max_entries:
                old_token, old_result = self._results.popitem(last=False)
                self._instance_tokens.pop(old_result.instance_id, None)
                evicted.append(old_token)

        if evicted:
            gc.collect()
        return {"token": token, "shape": shape, "fields": fields}

    def get_array(self, token: str, field: str) -> np.ndarray | None:
        """Return a cached field array, or ``None`` if it is unavailable."""
        with self._lock:
            result = self._results.get(token)
            if result is None:
                return None
            result.last_access = time.monotonic()
            self._results.move_to_end(token)
            array = result.arrays.get(field)
            if array is None:
                return None
            return array

    def get_result(self, token: str) -> CachedResult | None:
        """Return a cached result by token and mark it recently accessed."""
        with self._lock:
            result = self._results.get(token)
            if result is not None:
                result.last_access = time.monotonic()
                self._results.move_to_end(token)
            return result

    def get_result_for_instance(self, instance_id: str) -> CachedResult | None:
        """Return the latest cached result for an instance, if one exists."""
        with self._lock:
            token = self._instance_tokens.get(instance_id)
        if token is None:
            return None
        return self.get_result(token)

    def release_instance(self, instance_id: str) -> None:
        """Remove the cached result associated with an instance."""
        with self._lock:
            token = self._instance_tokens.pop(instance_id, None)
            if token is not None:
                self._results.pop(token, None)
        gc.collect()


result_cache = ResultCache()


def encode_array(x: np.ndarray) -> str:
    """
    Encode a NumPy array of float32 values as a base64 string.
    """
    return base64.b64encode(x.astype(np.float32, copy=False).tobytes()).decode("utf-8")


def decode_array(x_str: str, shape: list[int]) -> np.ndarray:
    """
    Decode a base64 string of float32 values into the requested shape.
    """
    img_bytes = base64.b64decode(x_str.encode("utf-8"))  # type: ignore
    flat_arr = np.frombuffer(img_bytes, dtype=np.float32)
    return flat_arr.reshape(shape)
