"""Shape-family-aware torch.compile dispatch helpers."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable

import torch

from tab_foundry.feature_types import metadata_has_explicit_feature_types
from tab_foundry.task_batching import task_batch_signature, task_batch_signature_text
from tab_foundry.types import TaskBatch

from .runtime import CompilePolicy, CudaGraphCapturePolicy
from .trainer_metrics import _compute_loss_and_metrics

SignatureFamily = tuple[int, int, int]
CudaGraphSignature = tuple[int, int, int, int, int | None, bool]
_SINGLE_TASK_TENSOR_DIMENSIONS = 2
_CUDA_GRAPH_WARMUP_STEPS = 3


def _signature_family(batch: TaskBatch) -> SignatureFamily:
    signature = task_batch_signature(batch)
    return int(signature[0]), int(signature[1]), int(signature[2])


def _signature_family_text(family: SignatureFamily) -> str:
    return f"{int(family[0])}x{int(family[1])}x{int(family[2])}"


def _task_batch_task_count(batch: TaskBatch) -> int:
    if batch.x_train.ndim == _SINGLE_TASK_TENSOR_DIMENSIONS:
        return 1
    return int(batch.x_train.shape[0])


def _cuda_graph_signature(batch: TaskBatch) -> CudaGraphSignature:
    n_train, n_test, n_features, num_classes = task_batch_signature(batch)
    return (
        _task_batch_task_count(batch),
        int(n_train),
        int(n_test),
        int(n_features),
        None if num_classes is None else int(num_classes),
        batch.feature_type_ids is not None,
    )


def _cuda_graph_signature_text(signature: CudaGraphSignature) -> str:
    task_count, n_train, n_test, n_features, num_classes, has_feature_type_ids = signature
    base_text = task_batch_signature_text((n_train, n_test, n_features, num_classes))
    feature_type_text = "feature_type_ids" if has_feature_type_ids else "no_feature_type_ids"
    return f"tasks{int(task_count)}:{base_text}:{feature_type_text}"


def _cuda_graph_fallback_reason(exc: Exception) -> str:
    message = str(exc).strip()
    return message if message else type(exc).__name__


def _clone_static_task_batch(batch: TaskBatch) -> TaskBatch:
    return TaskBatch(
        x_train=torch.empty_like(batch.x_train),
        y_train=torch.empty_like(batch.y_train),
        x_test=torch.empty_like(batch.x_test),
        y_test=torch.empty_like(batch.y_test),
        metadata=dict(batch.metadata),
        num_classes=batch.num_classes,
        feature_type_ids=(
            None if batch.feature_type_ids is None else torch.empty_like(batch.feature_type_ids)
        ),
    )


def _copy_task_batch_tensors(target: TaskBatch, source: TaskBatch) -> None:
    target.x_train.copy_(source.x_train)
    target.y_train.copy_(source.y_train)
    target.x_test.copy_(source.x_test)
    target.y_test.copy_(source.y_test)
    if target.feature_type_ids is not None and source.feature_type_ids is not None:
        target.feature_type_ids.copy_(source.feature_type_ids)


@dataclass(slots=True)
class _CudaGraphReplay:
    static_batch: TaskBatch
    captured_output: Any
    graph: torch.cuda.CUDAGraph
    task: str
    autocast_context_factory: Callable[[], AbstractContextManager[object]]

    def replay(self, batch: TaskBatch) -> tuple[torch.Tensor, dict[str, float]]:
        _copy_task_batch_tensors(self.static_batch, batch)
        self.graph.replay()
        with torch.no_grad():
            with self.autocast_context_factory():
                loss, metrics = _compute_loss_and_metrics(
                    self.captured_output,
                    self.static_batch,
                    task=self.task,
                )
        return loss.detach(), metrics


class SignatureFamilyCompileDispatcher:
    """Compile one callable per shape family and dispatch lazily at runtime."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        compile_policy: CompilePolicy,
        max_families: int,
        task: str,
        grad_accum_steps: int,
        cuda_graph_policy: CudaGraphCapturePolicy | None = None,
        cuda_graph_autocast_context_factory: (
            Callable[[], AbstractContextManager[object]] | None
        ) = None,
    ) -> None:
        if not compile_policy.enabled:
            raise ValueError("SignatureFamilyCompileDispatcher requires compile_policy.enabled=True")
        compile_fn = getattr(torch, "compile", None)
        if not callable(compile_fn):
            raise RuntimeError("runtime.compile_model=true requires torch.compile support")
        self._model = model
        self._compile_fn = compile_fn
        self._compile_kwargs = compile_policy.torch_compile_kwargs()
        self._max_families = int(max_families)
        self._task = str(task)
        self._grad_accum_steps = int(grad_accum_steps)
        self._compiled_by_family: dict[SignatureFamily, Any] = {}
        self._family_hits = 0
        self._family_misses = 0
        self._family_switches = 0
        self._uncached_family_calls = 0
        self._last_family: SignatureFamily | None = None
        self._family_call_counts: dict[str, int] = {}
        self._cuda_graph_policy = (
            cuda_graph_policy
            if cuda_graph_policy is not None
            else CudaGraphCapturePolicy(enabled=False, mode="off", max_families=0)
        )
        self._cuda_graph_autocast_context_factory = (
            cuda_graph_autocast_context_factory or nullcontext
        )
        self._cuda_graphs_by_signature: dict[CudaGraphSignature, _CudaGraphReplay] = {}
        self._cuda_graph_hits = 0
        self._cuda_graph_misses = 0
        self._cuda_graph_fallback_calls = 0
        self._cuda_graph_uncached_family_calls = 0
        self._cuda_graph_capture_failures = 0
        self._cuda_graph_call_counts: dict[str, int] = {}
        self._cuda_graph_failure_messages: dict[str, str] = {}

    def __call__(self, batch: TaskBatch):
        resolved_callable, _compiled = self._resolve_callable(batch)
        return resolved_callable(batch)

    def _resolve_callable(self, batch: TaskBatch) -> tuple[Callable[[TaskBatch], Any], bool]:
        family = _signature_family(batch)
        family_text = _signature_family_text(family)
        if self._last_family is not None and family != self._last_family:
            self._family_switches += 1
        self._last_family = family
        self._family_call_counts[family_text] = self._family_call_counts.get(family_text, 0) + 1

        compiled = self._compiled_by_family.get(family)
        if compiled is not None:
            self._family_hits += 1
            return compiled, True

        self._family_misses += 1
        if len(self._compiled_by_family) >= self._max_families:
            self._uncached_family_calls += 1
            return self._model, False

        compiled = self._compile_fn(self._model, **self._compile_kwargs)
        self._compiled_by_family[family] = compiled
        return compiled, True

    def _weighted_microstep_loss(self, loss: torch.Tensor, *, actual_task_count: int) -> torch.Tensor:
        resolved_task_count = int(actual_task_count)
        if resolved_task_count <= 0:
            raise RuntimeError(
                "task-batch accumulation requires a positive actual_task_count, "
                f"got {resolved_task_count}"
            )
        return loss * float(resolved_task_count) * float(self._grad_accum_steps)

    def _fallback_training_step(
        self,
        *,
        batch: TaskBatch,
        task: str,
        actual_task_count: int,
        accelerator: Any,
        resolved_callable: Callable[[TaskBatch], Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        with self._cuda_graph_autocast_context_factory():
            output = resolved_callable(batch)
            loss, metrics = _compute_loss_and_metrics(output, batch, task=task)
        accelerator.backward(
            self._weighted_microstep_loss(loss, actual_task_count=actual_task_count)
        )
        return loss, metrics

    def _build_cuda_graph_replay(
        self,
        *,
        compiled_callable: Callable[[TaskBatch], Any],
        batch: TaskBatch,
        task: str,
        actual_task_count: int,
    ) -> _CudaGraphReplay:
        if batch.x_train.device.type != "cuda":
            raise RuntimeError("CUDA graph capture requires CUDA device tensors")
        if actual_task_count != _task_batch_task_count(batch):
            raise RuntimeError(
                "CUDA graph capture requires a stable task-count signature, "
                f"expected {actual_task_count}, got {_task_batch_task_count(batch)}"
            )
        if batch.feature_type_ids is None and metadata_has_explicit_feature_types(batch.metadata):
            raise RuntimeError(
                "CUDA graph capture requires batch.feature_type_ids when explicit feature types are present"
            )

        static_batch = _clone_static_task_batch(batch)
        _copy_task_batch_tensors(static_batch, batch)
        self._model.zero_grad(set_to_none=True)
        stream = torch.cuda.Stream()
        current_stream = torch.cuda.current_stream()
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            for _ in range(_CUDA_GRAPH_WARMUP_STEPS):
                self._model.zero_grad(set_to_none=True)
                with self._cuda_graph_autocast_context_factory():
                    output = compiled_callable(static_batch)
                    loss, _metrics = _compute_loss_and_metrics(output, static_batch, task=task)
                self._weighted_microstep_loss(loss, actual_task_count=actual_task_count).backward()
        current_stream.wait_stream(stream)
        self._model.zero_grad(set_to_none=True)

        graph = torch.cuda.CUDAGraph()
        captured_output = None
        with torch.cuda.graph(graph):
            with self._cuda_graph_autocast_context_factory():
                captured_output = compiled_callable(static_batch)
                loss, _metrics = _compute_loss_and_metrics(captured_output, static_batch, task=task)
            self._weighted_microstep_loss(loss, actual_task_count=actual_task_count).backward()
        self._model.zero_grad(set_to_none=True)
        return _CudaGraphReplay(
            static_batch=static_batch,
            captured_output=captured_output,
            graph=graph,
            task=task,
            autocast_context_factory=self._cuda_graph_autocast_context_factory,
        )

    def training_step(
        self,
        *,
        batch: TaskBatch,
        task: str,
        actual_task_count: int,
        accelerator: Any,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        resolved_callable, compiled = self._resolve_callable(batch)
        if not self._cuda_graph_policy.enabled or not compiled:
            return self._fallback_training_step(
                batch=batch,
                task=task,
                actual_task_count=actual_task_count,
                accelerator=accelerator,
                resolved_callable=resolved_callable,
            )

        graph_signature = _cuda_graph_signature(batch)
        graph_signature_text = _cuda_graph_signature_text(graph_signature)
        self._cuda_graph_call_counts[graph_signature_text] = (
            self._cuda_graph_call_counts.get(graph_signature_text, 0) + 1
        )

        graph_replay = self._cuda_graphs_by_signature.get(graph_signature)
        if graph_replay is not None:
            self._cuda_graph_hits += 1
            return graph_replay.replay(batch)

        self._cuda_graph_misses += 1
        if len(self._cuda_graphs_by_signature) >= self._cuda_graph_policy.max_families:
            self._cuda_graph_uncached_family_calls += 1
            self._cuda_graph_fallback_calls += 1
            return self._fallback_training_step(
                batch=batch,
                task=task,
                actual_task_count=actual_task_count,
                accelerator=accelerator,
                resolved_callable=resolved_callable,
            )

        try:
            graph_replay = self._build_cuda_graph_replay(
                compiled_callable=resolved_callable,
                batch=batch,
                task=task,
                actual_task_count=actual_task_count,
            )
        except Exception as exc:
            self._cuda_graph_capture_failures += 1
            self._cuda_graph_fallback_calls += 1
            self._cuda_graph_failure_messages[graph_signature_text] = _cuda_graph_fallback_reason(
                exc
            )
            return self._fallback_training_step(
                batch=batch,
                task=task,
                actual_task_count=actual_task_count,
                accelerator=accelerator,
                resolved_callable=resolved_callable,
            )

        self._cuda_graphs_by_signature[graph_signature] = graph_replay
        return graph_replay.replay(batch)

    def summary(self) -> dict[str, Any]:
        compiled_families = sorted(_signature_family_text(family) for family in self._compiled_by_family)
        summary: dict[str, Any] = {
            "family_cache_hits": int(self._family_hits),
            "family_cache_misses": int(self._family_misses),
            "compiled_family_count": int(len(self._compiled_by_family)),
            "family_switch_count": int(self._family_switches),
            "uncached_family_call_count": int(self._uncached_family_calls),
            "compiled_families": compiled_families,
            "family_call_counts": dict(sorted(self._family_call_counts.items())),
            "cuda_graph_capture_mode": str(self._cuda_graph_policy.mode),
            "cuda_graph_max_families": int(self._cuda_graph_policy.max_families),
        }
        if not self._cuda_graph_policy.enabled:
            return summary
        summary.update(
            {
                "cuda_graph_cache_hits": int(self._cuda_graph_hits),
                "cuda_graph_cache_misses": int(self._cuda_graph_misses),
                "cuda_graph_captured_family_count": int(len(self._cuda_graphs_by_signature)),
                "cuda_graph_fallback_call_count": int(self._cuda_graph_fallback_calls),
                "cuda_graph_uncached_family_call_count": int(
                    self._cuda_graph_uncached_family_calls
                ),
                "cuda_graph_capture_failure_count": int(self._cuda_graph_capture_failures),
                "cuda_graph_captured_families": sorted(
                    _cuda_graph_signature_text(signature)
                    for signature in self._cuda_graphs_by_signature
                ),
                "cuda_graph_call_counts": dict(sorted(self._cuda_graph_call_counts.items())),
            }
        )
        if self._cuda_graph_failure_messages:
            summary["cuda_graph_failure_messages"] = dict(
                sorted(self._cuda_graph_failure_messages.items())
            )
        return summary
