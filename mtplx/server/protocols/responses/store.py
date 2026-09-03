"""Bounded, process-local ownership for Responses envelopes and lineage.

The registry stores JSON protocol state only. Generation lifecycle and model
cancellation remain owned by GenerationTurn and the server's existing
per-request cancellation handle.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


DEFAULT_MAX_ENTRIES = 256
DEFAULT_MAX_IN_FLIGHT = 64
DEFAULT_MAX_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_ENTRY_BYTES = 4 * 1024 * 1024
DEFAULT_IDLE_TTL_S = 3600.0
DEFAULT_TOMBSTONE_ENTRIES = 512


class ResponseStoreError(LookupError):
    """Structured lifecycle error suitable for the Responses wire."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: int,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = int(status_code)
        self.param = param

    def payload(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": "invalid_request_error",
                "param": self.param,
                "code": self.code,
            }
        }


@dataclass
class _StoredResponse:
    envelope: dict[str, Any]
    materialized_messages: list[dict[str, Any]]
    committed_at: float
    last_access_at: float
    size_bytes: int


@dataclass
class _InFlightResponse:
    response_id: str
    store: bool
    materialized_messages: list[dict[str, Any]]
    cancel: Callable[[], Any]
    started_at: float
    size_bytes: int
    cancel_requested: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    terminal_envelope: dict[str, Any] | None = None


@dataclass(frozen=True)
class _Tombstone:
    reason: str
    created_at: float


def _json_clone(value: Any) -> Any:
    """Copy and validate the JSON-only storage boundary."""

    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _json_size(*values: Any) -> int:
    return sum(
        len(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        for value in values
    )


def _positive_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


class ResponseRegistry:
    """Thread-safe bounded registry for local Responses protocol state."""

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_entry_bytes: int = DEFAULT_MAX_ENTRY_BYTES,
        idle_ttl_s: float = DEFAULT_IDLE_TTL_S,
        max_tombstones: int = DEFAULT_TOMBSTONE_ENTRIES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            min(
                max_entries,
                max_in_flight,
                max_bytes,
                max_entry_bytes,
                max_tombstones,
            )
            <= 0
        ):
            raise ValueError("response registry bounds must be positive")
        if idle_ttl_s <= 0:
            raise ValueError("response registry idle_ttl_s must be positive")
        self.max_entries = int(max_entries)
        self.max_in_flight = int(max_in_flight)
        self.max_bytes = int(max_bytes)
        self.max_entry_bytes = int(max_entry_bytes)
        self.idle_ttl_s = float(idle_ttl_s)
        self.max_tombstones = int(max_tombstones)
        self._clock = clock
        self._lock = threading.RLock()
        self._stored: OrderedDict[str, _StoredResponse] = OrderedDict()
        self._in_flight: dict[str, _InFlightResponse] = {}
        self._tombstones: OrderedDict[str, _Tombstone] = OrderedDict()
        self._bytes = 0
        self._in_flight_bytes = 0
        self._counters = {
            "committed_total": 0,
            "deleted_total": 0,
            "evicted_total": 0,
            "expired_total": 0,
            "cancel_requests_total": 0,
            "cancel_settled_total": 0,
        }

    @classmethod
    def from_env(cls) -> "ResponseRegistry":
        return cls(
            max_entries=_positive_env(
                "MTPLX_RESPONSE_STORE_MAX_ENTRIES", DEFAULT_MAX_ENTRIES
            ),
            max_in_flight=_positive_env(
                "MTPLX_RESPONSE_STORE_MAX_IN_FLIGHT", DEFAULT_MAX_IN_FLIGHT
            ),
            max_bytes=_positive_env(
                "MTPLX_RESPONSE_STORE_MAX_BYTES", DEFAULT_MAX_BYTES
            ),
            max_entry_bytes=_positive_env(
                "MTPLX_RESPONSE_STORE_MAX_ENTRY_BYTES", DEFAULT_MAX_ENTRY_BYTES
            ),
            idle_ttl_s=_positive_float_env(
                "MTPLX_RESPONSE_STORE_IDLE_TTL_S", DEFAULT_IDLE_TTL_S
            ),
            max_tombstones=_positive_env(
                "MTPLX_RESPONSE_STORE_MAX_TOMBSTONES",
                DEFAULT_TOMBSTONE_ENTRIES,
            ),
        )

    def allocate_id(self, preferred: str | None = None) -> str:
        """Return a collision-safe local id, retaining a safe unused hint."""

        with self._lock:
            self._prune_locked()
            if preferred and not self._known_locked(preferred):
                return preferred
            while True:
                candidate = f"resp_{uuid.uuid4().hex}"
                if not self._known_locked(candidate):
                    return candidate

    def begin(
        self,
        response_id: str,
        *,
        store: bool,
        materialized_messages: Sequence[Mapping[str, Any]],
        cancel: Callable[[], Any],
    ) -> None:
        messages = _json_clone(list(materialized_messages))
        size_bytes = _json_size(messages)
        now = self._clock()
        with self._lock:
            self._prune_locked(now)
            if self._known_locked(response_id):
                raise ResponseStoreError(
                    f"response id {response_id!r} already exists",
                    code="response_id_conflict",
                    status_code=409,
                )
            if size_bytes > self.max_entry_bytes:
                raise ResponseStoreError(
                    "response input exceeds the local lineage entry byte limit",
                    code="response_too_large",
                    status_code=413,
                    param="input",
                )
            if len(self._in_flight) >= self.max_in_flight:
                raise ResponseStoreError(
                    "response registry has reached its in-flight limit",
                    code="response_store_capacity",
                    status_code=429,
                )
            self._in_flight[response_id] = _InFlightResponse(
                response_id=response_id,
                store=bool(store),
                materialized_messages=messages,
                cancel=cancel,
                started_at=now,
                size_bytes=size_bytes,
            )
            self._in_flight_bytes += size_bytes

    def commit(
        self,
        response_id: str,
        envelope: Mapping[str, Any],
        *,
        materialized_messages: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        terminal = _json_clone(dict(envelope))
        messages = _json_clone(list(materialized_messages))
        now = self._clock()
        with self._lock:
            self._prune_locked(now)
            inflight = self._in_flight.pop(response_id, None)
            if inflight is None:
                stored = self._stored.get(response_id)
                if stored is not None:
                    return _json_clone(stored.envelope)
                return terminal
            self._in_flight_bytes -= inflight.size_bytes
            inflight.terminal_envelope = terminal
            inflight.done.set()
            tombstone = self._tombstones.get(response_id)
            if tombstone is not None and tombstone.reason == "deleted":
                return terminal
            if inflight.cancel_requested:
                self._counters["cancel_settled_total"] += 1
            if not inflight.store:
                return terminal
            size_bytes = _json_size(terminal, messages)
            if size_bytes > self.max_entry_bytes or size_bytes > self.max_bytes:
                self._add_tombstone_locked(response_id, "evicted", now)
                self._counters["evicted_total"] += 1
                return terminal
            self._stored[response_id] = _StoredResponse(
                envelope=terminal,
                materialized_messages=messages,
                committed_at=now,
                last_access_at=now,
                size_bytes=size_bytes,
            )
            self._bytes += size_bytes
            self._counters["committed_total"] += 1
            self._evict_pressure_locked(now)
            return _json_clone(terminal)

    def get(self, response_id: str) -> dict[str, Any]:
        now = self._clock()
        with self._lock:
            self._prune_locked(now)
            stored = self._stored.get(response_id)
            if stored is not None:
                stored.last_access_at = now
                return _json_clone(stored.envelope)
            self._raise_lookup_locked(response_id)
        raise AssertionError("unreachable")

    def parent_messages(self, response_id: str) -> list[dict[str, Any]]:
        now = self._clock()
        with self._lock:
            self._prune_locked(now)
            stored = self._stored.get(response_id)
            if stored is not None:
                stored.last_access_at = now
                return _json_clone(stored.materialized_messages)
            self._raise_lookup_locked(response_id, param="previous_response_id")
        raise AssertionError("unreachable")

    def request_cancel(self, response_id: str) -> _InFlightResponse:
        with self._lock:
            self._prune_locked()
            inflight = self._in_flight.get(response_id)
            if inflight is None:
                stored = self._stored.get(response_id)
                if stored is not None:
                    error = stored.envelope.get("error") or {}
                    if error.get("code") == "request_cancelled":
                        done = _InFlightResponse(
                            response_id=response_id,
                            store=True,
                            materialized_messages=stored.materialized_messages,
                            cancel=lambda: None,
                            started_at=stored.committed_at,
                            size_bytes=0,
                            cancel_requested=True,
                            terminal_envelope=_json_clone(stored.envelope),
                        )
                        done.done.set()
                        return done
                    raise ResponseStoreError(
                        f"response {response_id!r} is already terminal",
                        code="response_not_cancellable",
                        status_code=409,
                    )
                self._raise_lookup_locked(response_id)
            first_request = not inflight.cancel_requested
            inflight.cancel_requested = True
            if first_request:
                self._counters["cancel_requests_total"] += 1
            cancel = inflight.cancel if first_request else None
        if cancel is not None:
            cancel()
        return inflight

    @staticmethod
    def wait_terminal(
        inflight: _InFlightResponse, timeout_s: float
    ) -> dict[str, Any] | None:
        inflight.done.wait(max(0.0, timeout_s))
        envelope = inflight.terminal_envelope
        return _json_clone(envelope) if envelope is not None else None

    def delete(self, response_id: str) -> dict[str, Any]:
        cancel: Callable[[], Any] | None = None
        now = self._clock()
        with self._lock:
            self._prune_locked(now)
            tombstone = self._tombstones.get(response_id)
            if tombstone is not None and tombstone.reason == "deleted":
                return self._deleted_payload(response_id)
            stored = self._stored.pop(response_id, None)
            if stored is not None:
                self._bytes -= stored.size_bytes
            inflight = self._in_flight.get(response_id)
            if inflight is not None and not inflight.cancel_requested:
                inflight.cancel_requested = True
                cancel = inflight.cancel
                self._counters["cancel_requests_total"] += 1
            if stored is None and inflight is None:
                self._raise_lookup_locked(response_id)
            self._add_tombstone_locked(response_id, "deleted", now)
            self._counters["deleted_total"] += 1
        if cancel is not None:
            cancel()
        return self._deleted_payload(response_id)

    def is_in_flight(self, response_id: str) -> bool:
        with self._lock:
            return response_id in self._in_flight

    def terminal_for(self, response_id: str) -> dict[str, Any] | None:
        with self._lock:
            stored = self._stored.get(response_id)
            if stored is not None:
                return _json_clone(stored.envelope)
            inflight = self._in_flight.get(response_id)
            if inflight is not None and inflight.terminal_envelope is not None:
                return _json_clone(inflight.terminal_envelope)
            return None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._prune_locked()
            return {
                "entries": len(self._stored),
                "bytes": self._bytes,
                "in_flight": len(self._in_flight),
                "in_flight_bytes": self._in_flight_bytes,
                "tombstones": len(self._tombstones),
                "max_entries": self.max_entries,
                "max_in_flight": self.max_in_flight,
                "max_bytes": self.max_bytes,
                "max_entry_bytes": self.max_entry_bytes,
                "idle_ttl_s": self.idle_ttl_s,
                **self._counters,
            }

    def shutdown(self, timeout_s: float = 1.0) -> None:
        with self._lock:
            inflight = list(self._in_flight.values())
            callbacks = [
                item.cancel for item in inflight if not item.cancel_requested
            ]
            for item in inflight:
                item.cancel_requested = True
        for cancel in callbacks:
            try:
                cancel()
            except Exception:
                pass
        deadline = time.monotonic() + max(0.0, timeout_s)
        for item in inflight:
            item.done.wait(max(0.0, deadline - time.monotonic()))
        with self._lock:
            self._in_flight.clear()
            self._in_flight_bytes = 0

    def _known_locked(self, response_id: str) -> bool:
        return (
            response_id in self._stored
            or response_id in self._in_flight
            or response_id in self._tombstones
        )

    def _raise_lookup_locked(
        self, response_id: str, *, param: str | None = None
    ) -> None:
        if response_id in self._in_flight:
            raise ResponseStoreError(
                f"response {response_id!r} is still in progress",
                code="response_in_progress",
                status_code=409,
                param=param,
            )
        tombstone = self._tombstones.get(response_id)
        if tombstone is not None:
            raise ResponseStoreError(
                f"response {response_id!r} was {tombstone.reason}",
                code=f"response_{tombstone.reason}",
                status_code=410,
                param=param,
            )
        raise ResponseStoreError(
            f"response {response_id!r} was not found",
            code="response_not_found",
            status_code=404,
            param=param,
        )

    def _prune_locked(self, now: float | None = None) -> None:
        current = self._clock() if now is None else now
        expired = [
            response_id
            for response_id, stored in self._stored.items()
            if current - stored.last_access_at >= self.idle_ttl_s
        ]
        for response_id in expired:
            stored = self._stored.pop(response_id)
            self._bytes -= stored.size_bytes
            self._add_tombstone_locked(response_id, "evicted", current)
            self._counters["expired_total"] += 1
            self._counters["evicted_total"] += 1
        stale_tombstones = [
            response_id
            for response_id, tombstone in self._tombstones.items()
            if current - tombstone.created_at >= self.idle_ttl_s
        ]
        for response_id in stale_tombstones:
            self._tombstones.pop(response_id, None)

    def _evict_pressure_locked(self, now: float) -> None:
        while len(self._stored) > self.max_entries or self._bytes > self.max_bytes:
            response_id, stored = self._stored.popitem(last=False)
            self._bytes -= stored.size_bytes
            self._add_tombstone_locked(response_id, "evicted", now)
            self._counters["evicted_total"] += 1

    def _add_tombstone_locked(
        self, response_id: str, reason: str, now: float
    ) -> None:
        self._tombstones.pop(response_id, None)
        self._tombstones[response_id] = _Tombstone(reason=reason, created_at=now)
        while len(self._tombstones) > self.max_tombstones:
            self._tombstones.popitem(last=False)

    @staticmethod
    def _deleted_payload(response_id: str) -> dict[str, Any]:
        return {"id": response_id, "object": "response.deleted", "deleted": True}
