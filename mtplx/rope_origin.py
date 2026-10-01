"""The rotary origin a tensor-offset cache owns next to its logical offset.

Every compiled verify route carries the KV offset of a full-attention layer as
an int32 array (``TensorOffsetKVCache``, the two paged adapters in
``cache_state`` and the fixed QSA bank in ``graphbank``). A text request
rotates row ``i`` of a forward at ``offset + i``. An image request rotates it
at ``offset + rope_delta + i``: past the last image every position of the
request's table is the sequence index plus one constant
(``mtplx/vision/mrope.py``), so the whole table collapses to that delta for
every row a decode forward can write.

The delta shifts ROTARY positions only. ``offset`` keeps driving the masks,
the rollback window and every slice write, which are KV indices. Because the
origin rides the cache, no compiled route ever reads the request context: a
table or a delta read inside a verify trace would be baked into a graph that
every request of the process replays. The delta reaches the trace as a graph
input (``CompiledVerifyBank._make_verify_step``) or as an implicit input
through ``compile_state`` (the graph bank and the device draft cores), never
as a Python constant.

``as_rope_delta`` is the one validator of that value; ``rope_offset_of`` is
the one place the attention routes read the origin from.
"""

from __future__ import annotations

from typing import Any


def as_rope_delta(value: Any):
    """The rotary delta of an image request as one int32 value, or None.

    The delta is a graph input of the compiled verify step and the position
    operand of the rope kernels, whose contract is one int32 value
    (``kernels/qwen4_m4_rope._as_i32_scalar``; ``mx.fast.rope`` takes the
    same). Anything else is refused here, where the request is admitted,
    instead of inside a trace. A host integer becomes a one-element array;
    an array is checked, never read, so a tracer passes through.
    """

    if value is None:
        return None
    import mlx.core as mx

    if isinstance(value, mx.array):
        if value.dtype != mx.int32 or int(value.size) != 1:
            raise TypeError(
                "rope_delta must be one int32 value; got "
                f"dtype={value.dtype}, shape={tuple(value.shape)}"
            )
        return value.reshape((1,))
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"rope_delta must be an int or a one-element int32 array; got {value!r}"
        )
    return mx.array([int(value)], dtype=mx.int32)


class RotaryOrigin:
    """Mixin for a container whose logical offset lives at ``self.cache[2]``.

    ``rope_delta`` is None for a text request and one int32 value for an
    image request. ``rope_offset`` is the rotary position of the next row the
    container writes: for a text request it IS ``offset``, the same array
    object, so a text verify trace holds exactly the nodes it held before
    image requests could reach the compiled routes.

    The delta lives in ONE place, ``rope_state``: the list a compiled closure
    captures next to the cache leaves (``compile_state``). ``mx.compile``
    swaps a captured list's slots for tracers while it traces and reads the
    slots again on every call, so a delta must be read through the list and
    nowhere else: a second reference to the same array is an uncaptured
    input to the trace, and a delta stamped after the closure was built is
    only seen through the slot. The list object itself is never replaced,
    for the same reason.
    """

    def _init_rotary_origin(self, rope_delta: Any = None) -> None:
        self.rope_state: list[Any] = []
        self.rope_delta = rope_delta

    @property
    def rope_delta(self):
        state = getattr(self, "rope_state", None)
        return state[0] if state else None

    @rope_delta.setter
    def rope_delta(self, value) -> None:
        delta = as_rope_delta(value)
        state = getattr(self, "rope_state", None)
        if state is None:
            self.rope_state = state = []
        state[:] = [] if delta is None else [delta]

    @property
    def rope_offset(self):
        offset = self.cache[2]
        state = self.rope_state
        if not state:
            return offset
        return offset + state[0]


def cache_owns_rotary_origin(cache: Any) -> bool:
    """True for a tensor-offset cache stamped with an image request's delta."""

    return getattr(cache, "rope_delta", None) is not None


def note_unowned_rotary_origin(cache: Any) -> None:
    """The routing-hole canary, at the one place the cache is known.

    A tensor-offset cache reached by an armed image request WITHOUT a delta
    is a compiled route that promoted the cache without the admission
    handing it the delta. Its rows keep the stock positions; the dense
    adapter's ledger counts the call (``dense_mrope.note_unowned_tensor_offset``,
    which raises under ``MTPLX_DENSE_MROPE_STRICT=1``), never silent. A
    stock cache, a stamped cache and a text request cost one attribute read.
    """

    if hasattr(cache, "rope_offset") and getattr(cache, "rope_delta", None) is None:
        from .dense_mrope import note_unowned_tensor_offset

        note_unowned_tensor_offset()


def rope_offset_of(cache: Any):
    """The position an attention route ropes this cache's next row at.

    A stock cache has no origin of its own: its ``offset`` (a host integer)
    is handed to the layer's rope, where the dense image position adapter
    resolves it through the request's table. A tensor-offset cache owns its
    origin (``RotaryOrigin.rope_offset``): the plain offset array for a text
    request, ``offset + rope_delta`` for an image request (and the canary
    above for a delta it should have had).
    """

    origin = getattr(cache, "rope_offset", None)
    if origin is None:
        return cache.offset
    note_unowned_rotary_origin(cache)
    return origin


def stamp_rope_delta(cache: Any, rope_delta: Any) -> int:
    """Give every rotary-origin container of ``cache`` the request's delta.

    Returns the number of containers stamped. Called after every promotion
    of a request's caches, so a container promoted mid-generation (a growth
    step, a device draft core) rotates where the earlier ones do.
    """

    delta = as_rope_delta(rope_delta)
    stamped = 0
    for entry in cache or []:
        if hasattr(entry, "rope_offset") and hasattr(entry, "rope_delta"):
            entry.rope_delta = delta
            stamped += 1
    return stamped
