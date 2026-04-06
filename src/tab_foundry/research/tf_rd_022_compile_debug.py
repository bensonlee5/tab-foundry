"""Short compile-debug ladder for TF-RD-022."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Sequence

from omegaconf import OmegaConf

from tab_foundry.repo_paths import repo_root
from tab_foundry.training.trainer import train

from .tf_rd_022_compile_first import (
    TF_RD_022_COMPILE_FIRST_EXPERIMENT,
    _compose_tf_rd_022_compile_cfg,
)

_TORCH_LOGS_VALUE = "graph_breaks,recompiles,dynamic,guards"
_SUMMARY_FILENAME = "compile_debug_summary.json"
_REQUEST_FILENAME = "variant_request.json"
_RESOLVED_CONFIG_FILENAME = "resolved_config.json"
_RESULT_FILENAME = "subprocess_result.json"
_STDOUT_FILENAME = "stdout.log"
_STDERR_FILENAME = "stderr.log"
_RUN_OUTPUT_DIRNAME = "run"
_TORCH_TRACE_DIRNAME = "torch_trace"
_GRAPH_BREAK_RE = re.compile(r"(?im)graph break[^\n]*?\bat\s+((?:[A-Za-z]:)?\S+?\.py:\d+)")
_RECOMPILE_RE = re.compile(r"(?im)\brecompil(?:e|ing|es)\b")
_AUTOTUNE_RE = re.compile(r"(?im)\bautotune\b")


@dataclass(frozen=True, slots=True)
class CompileDebugVariant:
    """One compile-debug ladder variant."""

    name: str
    compile_model: bool
    compile_backend: str
    compile_mode: str


TF_RD_022_COMPILE_DEBUG_VARIANTS = (
    CompileDebugVariant(
        name="baseline_uncompiled",
        compile_model=False,
        compile_backend="inductor",
        compile_mode="max-autotune-no-cudagraphs",
    ),
    CompileDebugVariant(
        name="compile_eager",
        compile_model=True,
        compile_backend="eager",
        compile_mode="max-autotune-no-cudagraphs",
    ),
    CompileDebugVariant(
        name="compile_aot_eager",
        compile_model=True,
        compile_backend="aot_eager",
        compile_mode="max-autotune-no-cudagraphs",
    ),
    CompileDebugVariant(
        name="compile_inductor_default",
        compile_model=True,
        compile_backend="inductor",
        compile_mode="default",
    ),
    CompileDebugVariant(
        name="compile_inductor_max_autotune",
        compile_model=True,
        compile_backend="inductor",
        compile_mode="max-autotune-no-cudagraphs",
    ),
)


def _variant_by_name(name: str) -> CompileDebugVariant:
    normalized_name = str(name).strip()
    for variant in TF_RD_022_COMPILE_DEBUG_VARIANTS:
        if variant.name == normalized_name:
            return variant
    raise ValueError(
        f"unknown TF-RD-022 compile debug variant {name!r}; "
        f"expected one of {[variant.name for variant in TF_RD_022_COMPILE_DEBUG_VARIANTS]}"
    )


def _variant_dir(root: Path, variant: CompileDebugVariant) -> Path:
    return root / variant.name


def _run_output_dir(variant_dir: Path) -> Path:
    return variant_dir / _RUN_OUTPUT_DIRNAME


def _torch_trace_dir(variant_dir: Path) -> Path:
    return variant_dir / _TORCH_TRACE_DIRNAME


def _request_payload(
    *,
    variant: CompileDebugVariant,
    variant_dir: Path,
    max_steps: int,
) -> dict[str, Any]:
    return {
        "variant": asdict(variant),
        "variant_dir": str(variant_dir),
        "run_output_dir": str(_run_output_dir(variant_dir)),
        "max_steps": int(max_steps),
        "experiment": TF_RD_022_COMPILE_FIRST_EXPERIMENT,
    }


def _build_variant_requests(
    output_dir: Path | str,
    *,
    max_steps: int,
    variant_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    suite_output_dir = Path(str(output_dir)).expanduser().resolve()
    selected_variants = (
        [_variant_by_name(name) for name in variant_names]
        if variant_names is not None
        else list(TF_RD_022_COMPILE_DEBUG_VARIANTS)
    )
    return [
        _request_payload(
            variant=variant,
            variant_dir=_variant_dir(suite_output_dir, variant),
            max_steps=max_steps,
        )
        for variant in selected_variants
    ]


def _build_variant_env(variant_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["TORCH_LOGS"] = _TORCH_LOGS_VALUE
    env["TORCH_TRACE"] = str(_torch_trace_dir(variant_dir))
    return env


def _build_variant_command(
    *,
    python_executable: str,
    request_path: Path,
) -> list[str]:
    return [
        python_executable,
        "-c",
        (
            "from tab_foundry.research.tf_rd_022_compile_debug import "
            "_run_variant_subprocess_request; "
            "import sys; "
            "raise SystemExit(_run_variant_subprocess_request(sys.argv[1]))"
        ),
        str(request_path),
    ]


def tf_rd_022_compile_debug_variant_cfg(
    *,
    variant_name: str,
    output_dir: Path | str,
    max_steps: int = 24,
) -> Any:
    """Resolve one TF-RD-022 compile-debug ladder variant config."""

    variant = _variant_by_name(variant_name)
    return _compose_tf_rd_022_compile_cfg(
        output_dir=output_dir,
        max_steps=max_steps,
        eval_every=max_steps,
        checkpoint_every=max_steps,
        compile_model=variant.compile_model,
        compile_backend=variant.compile_backend,
        compile_mode=variant.compile_mode,
        run_name_suffix=f"-{variant.name}",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_variant_subprocess_request(request_path: str | Path) -> int:
    request = json.loads(Path(str(request_path)).read_text(encoding="utf-8"))
    variant = _variant_by_name(str(request["variant"]["name"]))
    variant_dir = Path(str(request["variant_dir"])).expanduser().resolve()
    run_output_dir = Path(str(request["run_output_dir"])).expanduser().resolve()
    cfg = tf_rd_022_compile_debug_variant_cfg(
        variant_name=variant.name,
        output_dir=run_output_dir,
        max_steps=int(request["max_steps"]),
    )
    _write_json(
        variant_dir / _RESOLVED_CONFIG_FILENAME,
        OmegaConf.to_container(cfg, resolve=True),
    )
    result = train(cfg)
    _write_json(
        variant_dir / _RESULT_FILENAME,
        {
            "output_dir": str(result.output_dir),
            "global_step": int(result.global_step),
            "best_checkpoint": None
            if result.best_checkpoint is None
            else str(result.best_checkpoint),
        },
    )
    return 0


def _parse_compile_debug_log(log_text: str) -> dict[str, Any]:
    graph_break_locations = _GRAPH_BREAK_RE.findall(log_text)
    return {
        "graph_break_count": len(graph_break_locations),
        "graph_break_locations": sorted({str(location) for location in graph_break_locations}),
        "recompile_count": len(_RECOMPILE_RE.findall(log_text)),
        "autotune_count": len(_AUTOTUNE_RE.findall(log_text)),
    }


def _artifact_presence(variant_dir: Path, *, run_output_dir: Path) -> dict[str, bool]:
    return {
        "request": (variant_dir / _REQUEST_FILENAME).exists(),
        "resolved_config": (variant_dir / _RESOLVED_CONFIG_FILENAME).exists(),
        "subprocess_result": (variant_dir / _RESULT_FILENAME).exists(),
        "stdout": (variant_dir / _STDOUT_FILENAME).exists(),
        "stderr": (variant_dir / _STDERR_FILENAME).exists(),
        "torch_trace": _torch_trace_dir(variant_dir).exists(),
        "training_surface_record": (run_output_dir / "training_surface_record.json").exists(),
        "train_history_jsonl": (run_output_dir / "train_history.jsonl").exists(),
    }


def _variant_summary(
    *,
    request: dict[str, Any],
    return_code: int,
    wall_time_seconds: float,
) -> dict[str, Any]:
    variant_dir = Path(str(request["variant_dir"])).expanduser().resolve()
    run_output_dir = Path(str(request["run_output_dir"])).expanduser().resolve()
    stdout_text = (variant_dir / _STDOUT_FILENAME).read_text(encoding="utf-8")
    stderr_text = (variant_dir / _STDERR_FILENAME).read_text(encoding="utf-8")
    parsed_logs = _parse_compile_debug_log(stdout_text + "\n" + stderr_text)
    return {
        "name": str(request["variant"]["name"]),
        "wall_time_seconds": float(wall_time_seconds),
        "return_code": int(return_code),
        "compile": {
            "compile_model": bool(request["variant"]["compile_model"]),
            "compile_backend": str(request["variant"]["compile_backend"]),
            "compile_mode": str(request["variant"]["compile_mode"]),
        },
        **parsed_logs,
        "variant_dir": str(variant_dir),
        "run_output_dir": str(run_output_dir),
        "stdout_path": str(variant_dir / _STDOUT_FILENAME),
        "stderr_path": str(variant_dir / _STDERR_FILENAME),
        "artifacts": _artifact_presence(variant_dir, run_output_dir=run_output_dir),
    }


def run_tf_rd_022_compile_debug_suite(
    output_dir: Path | str,
    *,
    max_steps: int = 24,
    python_executable: str | Path | None = None,
    variant_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the short TF-RD-022 compile debug ladder in clean subprocesses."""

    suite_output_dir = Path(str(output_dir)).expanduser().resolve()
    suite_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_python = str(python_executable or sys.executable)
    variants_summary: list[dict[str, Any]] = []
    for request in _build_variant_requests(
        suite_output_dir,
        max_steps=max_steps,
        variant_names=variant_names,
    ):
        variant_dir = Path(str(request["variant_dir"])).expanduser().resolve()
        variant_dir.mkdir(parents=True, exist_ok=True)
        request_path = variant_dir / _REQUEST_FILENAME
        _write_json(request_path, request)
        started_at = time.perf_counter()
        completed = subprocess.run(
            _build_variant_command(
                python_executable=resolved_python,
                request_path=request_path,
            ),
            cwd=str(repo_root()),
            env=_build_variant_env(variant_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        wall_time_seconds = time.perf_counter() - started_at
        (variant_dir / _STDOUT_FILENAME).write_text(completed.stdout, encoding="utf-8")
        (variant_dir / _STDERR_FILENAME).write_text(completed.stderr, encoding="utf-8")
        variants_summary.append(
            _variant_summary(
                request=request,
                return_code=completed.returncode,
                wall_time_seconds=wall_time_seconds,
            )
        )
    summary = {
        "experiment": TF_RD_022_COMPILE_FIRST_EXPERIMENT,
        "output_dir": str(suite_output_dir),
        "max_steps": int(max_steps),
        "python_executable": resolved_python,
        "torch_logs": _TORCH_LOGS_VALUE,
        "variants": variants_summary,
    }
    _write_json(suite_output_dir / _SUMMARY_FILENAME, summary)
    return summary
