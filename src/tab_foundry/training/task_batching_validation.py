"""Preflight validation for task batching against model capabilities."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator

from torch.utils.data import Dataset

from tab_foundry.data.dataset import PackedParquetTaskDataset, TaskSignature
from tab_foundry.model.architectures.tabfoundry_staged.resolved import resolve_staged_surface
from tab_foundry.model.spec import ModelBuildSpec, STAGED_MODEL_ARCH
from tab_foundry.task_batching import task_batch_signature_text
from tab_foundry.types import TaskBatch


def _iter_signature_bucket_sizes(
    dataset: PackedParquetTaskDataset,
    *,
    shuffle: bool,
) -> Iterator[tuple[TaskSignature, int]]:
    if shuffle:
        bucket_counts: OrderedDict[TaskSignature, int] = OrderedDict()
        for index in range(len(dataset)):
            signature = dataset.task_signature(index)
            bucket_counts[signature] = bucket_counts.get(signature, 0) + 1
        yield from bucket_counts.items()
        return

    current_signature: TaskSignature | None = None
    current_count = 0
    for index in range(len(dataset)):
        signature = dataset.task_signature(index)
        if current_signature is None or signature == current_signature:
            current_signature = signature
            current_count += 1
            continue
        yield current_signature, current_count
        current_signature = signature
        current_count = 1
    if current_signature is not None and current_count > 0:
        yield current_signature, current_count


def validate_task_batching_support(
    dataset: Dataset[TaskBatch],
    *,
    task_batch_size: int,
    model_spec: ModelBuildSpec,
    shuffle: bool,
    context: str,
) -> None:
    """Reject task batching combinations that the current model path cannot execute."""

    resolved_task_batch_size = int(task_batch_size)
    if resolved_task_batch_size <= 1:
        return
    if model_spec.arch != STAGED_MODEL_ARCH:
        return
    if not isinstance(dataset, PackedParquetTaskDataset):
        return

    surface = resolve_staged_surface(model_spec)
    if surface.head != "many_class":
        return

    many_class_base = int(model_spec.many_class_base)
    for signature, bucket_size in _iter_signature_bucket_sizes(dataset, shuffle=shuffle):
        num_classes = signature[3]
        if num_classes is None or int(num_classes) <= many_class_base:
            continue
        emitted_batch_size = min(int(bucket_size), resolved_task_batch_size)
        if emitted_batch_size <= 1:
            continue
        raise RuntimeError(
            f"{context} with task_batch_size={resolved_task_batch_size} supports staged "
            "many_class low-class batches and singleton true-many-class fallbacks, but "
            "tensor-batched true-many-class execution is deferred to the later many-class "
            "implementation. "
            f"Offending signature={task_batch_signature_text(signature)}, "
            f"bucket_size={int(bucket_size)}, emitted_batch_size={int(emitted_batch_size)}, "
            f"many_class_base={many_class_base}, shuffle={bool(shuffle)}."
        )
