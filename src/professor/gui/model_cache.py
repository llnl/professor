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
    key: str
    helper: ProfessorHelper
    ref_count: int = 0
    last_access: float = 0.0


@dataclass
class CachedResult:
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
        self._config = prof_config
        self._device: torch.device | str = ""
        self._model: torch.nn.Module | None = None

    def close(self) -> None:
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
    def __init__(self, max_entries: int = DEFAULT_CACHE_ENTRIES) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[str, CachedModel] = OrderedDict()
        self._session_keys: dict[str, str] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    def release_session(self, session_id: str) -> None:
        with self._lock:
            old_key = self._session_keys.pop(session_id, None)
            if old_key is None:
                return
            entry = self._entries.get(old_key)
            if entry is not None:
                entry.ref_count = max(0, entry.ref_count - 1)
        self._evict_if_needed()

    def get_or_load(self, payload: dict[str, Any], session_id: str) -> ProfessorHelper:
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
        with self._lock:
            entry = self._entries.get(cache_key)
            if entry is not None:
                entry.last_access = time.monotonic()
                self._entries.move_to_end(cache_key)
            return entry

    def _get_key_lock(self, cache_key: str) -> threading.Lock:
        with self._lock:
            lock = self._locks.get(cache_key)
            if lock is None:
                lock = threading.Lock()
                self._locks[cache_key] = lock
            return lock

    def _load_entry(self, payload: dict[str, Any]) -> CachedModel:
        prof_config = config.Config.from_text(payload["yaml_text"], source=payload["source"])
        helper = ProfessorHelper(prof_config)
        return CachedModel(key=payload["cache_key"], helper=helper, last_access=time.monotonic())

    def _release_previous_locked(self, previous_key: str | None) -> None:
        if previous_key is None:
            return
        previous_entry = self._entries.get(previous_key)
        if previous_entry is None:
            return
        previous_entry.ref_count = max(0, previous_entry.ref_count - 1)

    def _evict_if_needed(self) -> None:
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
    def __init__(self, max_entries: int = DEFAULT_RESULT_CACHE_ENTRIES) -> None:
        self._max_entries = max_entries
        self._results: OrderedDict[str, CachedResult] = OrderedDict()
        self._instance_tokens: dict[str, str] = {}
        self._lock = threading.Lock()

    def put(self, instance_id: str, fields: list[str], output: np.ndarray) -> dict[str, Any]:
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
        with self._lock:
            result = self._results.get(token)
            if result is not None:
                result.last_access = time.monotonic()
                self._results.move_to_end(token)
            return result

    def get_result_for_instance(self, instance_id: str) -> CachedResult | None:
        with self._lock:
            token = self._instance_tokens.get(instance_id)
        if token is None:
            return None
        return self.get_result(token)

    def release_instance(self, instance_id: str) -> None:
        with self._lock:
            token = self._instance_tokens.pop(instance_id, None)
            if token is not None:
                self._results.pop(token, None)
        gc.collect()


result_cache = ResultCache()


def encode_array(x: np.ndarray) -> str:
    """
    Encode an NDarray as a string.
    """
    return base64.b64encode(x.astype(np.float32, copy=False).tobytes()).decode("utf-8")


def decode_array(x_str: str, shape: list[int]) -> np.ndarray:
    """
    Decode an NDarray from a string.
    """
    img_bytes = base64.b64decode(x_str.encode("utf-8"))  # type: ignore
    flat_arr = np.frombuffer(img_bytes, dtype=np.float32)
    return flat_arr.reshape(shape)
