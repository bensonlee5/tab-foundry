"""Print architecture summary of the promoted anchor model."""

from __future__ import annotations

import torch
from torch import nn
import torchinfo

from tab_foundry.research.promoted_bridge_baseline import (
    build_promoted_bridge_baseline_model,
    promoted_bridge_baseline_payload,
)
class _TorchinfoClassifierWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
    ) -> torch.Tensor:
        forward_batched = getattr(self.model, "forward_batched", None)
        if not callable(forward_batched):
            raise RuntimeError("visualization wrapper requires a model with forward_batched()")
        x_all = torch.cat([x_train.unsqueeze(0), x_test.unsqueeze(0)], dim=1)
        y_train_batch = y_train.unsqueeze(0)
        return forward_batched(
            x_all=x_all,
            y_train=y_train_batch,
            train_test_split_index=int(x_train.shape[0]),
        )


def _summary_inputs(*, feature_count: int = 6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_train = torch.randn(4, feature_count, dtype=torch.float32)
    y_train = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    x_test = torch.randn(2, feature_count, dtype=torch.float32)
    return x_train, y_train, x_test


def main() -> None:
    payload = promoted_bridge_baseline_payload()
    model = build_promoted_bridge_baseline_model()
    wrapper = _TorchinfoClassifierWrapper(model)
    x_train, y_train, x_test = _summary_inputs()

    print("=" * 80)
    print("PROMOTED BRIDGE BASELINE")
    print("=" * 80)
    print(
        f"stage_label={payload['model']['stage_label']} "
        f"input_normalization={payload['model']['input_normalization']} "
        f"row_pool={payload['model']['module_overrides']['row_pool']}"
    )
    print(
        f"training={payload['training']['surface_label']} "
        f"lr_max={payload['training']['schedule_stage']['lr_max']:.6f} "
        f"min_lr={payload['training']['optimizer_min_lr']:.6f}"
    )
    print()
    print("=" * 80)
    print("TORCHINFO SUMMARY")
    print("=" * 80)
    torchinfo.summary(
        wrapper,
        input_data=[x_train, y_train, x_test],
        col_names=["num_params", "params_percent", "trainable"],
        col_width=18,
        depth=5,
        verbose=1,
    )

    print()
    print("=" * 80)
    print("PER-SUBMODULE PARAM COUNTS")
    print("=" * 80)
    total = 0
    for name, child in model.named_children():
        n_params = sum(p.numel() for p in child.parameters())
        total += n_params
        print(f"  {name:<40s} {n_params:>10,}")
    print(f"  {'TOTAL':<40s} {total:>10,}")


if __name__ == "__main__":
    main()
