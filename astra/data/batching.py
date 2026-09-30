"""Bounded CPU/device preparation overlapping the current batch's model work."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
import time

import torch


@dataclass(frozen=True)
class PreparedBatchResult:
    value: object
    main_wait_seconds: float
    cpu_prepare_seconds: float
    materialize_host_seconds: float


def record_stream(value, stream):
    """Protect nested CUDA storage until its consuming stream finishes using it."""
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            value.record_stream(stream)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in dataclass_fields(value):
            record_stream(getattr(value, field.name), stream)
    elif isinstance(value, dict):
        for item in value.values():
            record_stream(item, stream)
    elif isinstance(value, (tuple, list)):
        for item in value:
            record_stream(item, stream)


@contextmanager
def prefetched_batches(requests, prepare_cpu, encode, *, enabled=True, pipeline="cpu", device=None):
    """Yield ordered packets with at most one CPU future and one next GPU packet.

    ``prepare_cpu(request)`` may read caches and generate deterministic CPU masks;
    it must not use global RNG state or launch CUDA operations. In ``cpu`` mode,
    ``encode(prepared)`` runs on the consuming thread/current stream. ``device``
    mode with a CUDA device uses one GPU worker and one separate CUDA stream.
    Existing GPU inputs (such as epoch masks) must be ready on the caller's current
    stream before entering this context. Encode must not use global device RNG.
    The consumer must use yielded tensors on that same main stream; transferring
    them to another stream requires the caller's usual wait/record_stream handling.

    A ready event orders each GPU packet before main-stream use. Recursive stream
    recording protects both GPU inputs and returned tensors; pinned CPU buffers
    use PyTorch's asynchronous transfer lifetime tracking. CPU devices fall back
    to CPU mode, while ``enabled=False`` disables both workers.

    Closing joins GPU then CPU workers and finally synchronizes the prefetch stream.
    No epoch-sized count/feature cache or unbounded request queue is created.
    """
    if pipeline not in ("cpu", "device"):
        raise ValueError("batch pipeline must be 'cpu' or 'device'")
    if pipeline == "device" and device is None:
        raise ValueError("device prefetch requires an explicit device")
    device = None if device is None else torch.device(device)
    use_device = enabled and pipeline == "device" and device.type == "cuda"
    grad_enabled = torch.is_grad_enabled()
    inference_enabled = torch.is_inference_mode_enabled()
    autocast_enabled = torch.is_autocast_enabled("cuda") if use_device else False
    autocast_dtype = torch.get_autocast_dtype("cuda") if use_device else None
    source = iter(requests)
    main_stream = torch.cuda.current_stream(device) if use_device else None
    prefetch_stream = torch.cuda.Stream(device=device) if use_device else None
    if prefetch_stream is not None:
        # Pre-created GPU masks belong to the main stream.
        prefetch_stream.wait_stream(main_stream)
    cpu_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="v033_cpu") if enabled else None
    gpu_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="v033_gpu") if use_device else None
    cpu_pending = None
    sentinel = object()

    def prepare(request):
        started = time.monotonic()
        value = prepare_cpu(request)
        return value, time.monotonic() - started

    def consume_cpu():
        request = next(source, sentinel)
        if request is sentinel:
            return
        pending = cpu_worker.submit(prepare, request) if cpu_worker is not None else None
        while request is not sentinel:
            started = time.monotonic()
            prepared, cpu_seconds = pending.result() if cpu_worker is not None else prepare(request)
            wait_seconds = time.monotonic() - started
            request = next(source, sentinel)
            pending = (cpu_worker.submit(prepare, request)
                       if cpu_worker is not None and request is not sentinel else None)
            started = time.monotonic()
            value = encode(prepared)
            materialize_seconds = time.monotonic() - started
            del prepared
            yield PreparedBatchResult(value, wait_seconds, cpu_seconds, materialize_seconds)
            # Release the consumed device packet before materializing another batch.
            del value

    def prepare_device():
        # Only the GPU worker advances this future chain and request iterator.
        nonlocal cpu_pending
        prepared, cpu_seconds = cpu_pending.result()
        cpu_pending = None
        request = next(source, sentinel)
        has_next = request is not sentinel
        if has_next:
            cpu_pending = cpu_worker.submit(prepare, request)
        with torch.cuda.device(device), torch.cuda.stream(prefetch_stream), \
             torch.inference_mode(inference_enabled), torch.set_grad_enabled(grad_enabled), \
             torch.autocast("cuda", enabled=autocast_enabled, dtype=autocast_dtype):
            record_stream(prepared, prefetch_stream)
            started = time.monotonic()
            value = encode(prepared)
            materialize_seconds = time.monotonic() - started
            ready = torch.cuda.Event()
            ready.record(prefetch_stream)
        del prepared
        return value, cpu_seconds, materialize_seconds, ready, has_next

    def consume_device():
        nonlocal cpu_pending
        request = next(source, sentinel)
        if request is sentinel:
            return
        cpu_pending = cpu_worker.submit(prepare, request)
        pending = gpu_worker.submit(prepare_device)
        while pending is not None:
            started = time.monotonic()
            value, cpu_seconds, materialize_seconds, ready, has_next = pending.result()
            wait_seconds = time.monotonic() - started
            # Replacing the completed future releases its retained result promptly.
            pending = gpu_worker.submit(prepare_device) if has_next else None
            main_stream.wait_event(ready)
            record_stream(value, main_stream)
            del ready
            yield PreparedBatchResult(value, wait_seconds, cpu_seconds, materialize_seconds)
            del value

    iterator = consume_device() if use_device else consume_cpu()
    try:
        yield iterator
    finally:
        iterator.close()
        # The GPU worker may still request a CPU batch; join it before the CPU pool.
        try:
            if gpu_worker is not None:
                gpu_worker.shutdown(wait=True, cancel_futures=True)
        finally:
            try:
                if cpu_worker is not None:
                    cpu_worker.shutdown(wait=True, cancel_futures=True)
            finally:
                cpu_pending = None
                if prefetch_stream is not None:
                    prefetch_stream.synchronize()
