"""Shared click helpers for the packaged CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import click

from tab_foundry.data.corpus_materialization import default_materialize_processes
from tab_foundry.research.sweep import paths_io as sweep_paths


GROUP_KWARGS: dict[str, Any] = {
    "invoke_without_command": False,
    "no_args_is_help": False,
}

DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
_UINT32_MAX = 4_294_967_295

ClickDecorator = Callable[[Any], Any]


def run_click_command(
    command: click.Command,
    argv: Sequence[str] | None = None,
    *,
    prog_name: str | None = None,
) -> int:
    try:
        result = command.main(
            args=None if argv is None else list(argv),
            prog_name=prog_name,
            standalone_mode=False,
        )
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from exc
    except click.Abort as exc:
        raise SystemExit(1) from exc
    if result is None:
        return 0
    return int(result)


def apply_click_decorators(*decorators: ClickDecorator) -> ClickDecorator:
    """Apply click decorators in the same top-to-bottom order as stacked syntax."""

    def _decorator(func: Any) -> Any:
        for decorator in reversed(decorators):
            func = decorator(func)
        return func

    return _decorator


def emit_payload(
    payload: Any,
    *,
    json_mode: bool,
    render_text: Callable[[Any], str] | None = None,
) -> None:
    if json_mode:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    if render_text is None:
        raise ValueError("render_text is required when json_mode is false")
    click.echo(render_text(payload))


def json_output_option(func: Any) -> Any:
    return click.option(
        "--json",
        "json_mode",
        is_flag=True,
        help="Emit machine-readable JSON",
    )(func)


def path_option(
    name: str,
    *,
    help: str,
    default: str | Path | None = None,
    required: bool = False,
) -> ClickDecorator:
    option_kwargs: dict[str, Any] = {
        "required": required,
        "type": click.Path(path_type=Path),
        "help": help,
    }
    if default is not None:
        option_kwargs["default"] = str(default)
        option_kwargs["show_default"] = True
    return click.option(f"--{name}", **option_kwargs)


def device_option(
    *,
    default: str | None = "auto",
    choices: Sequence[str] = DEVICE_CHOICES,
    help: str = "Execution device; defaults to auto",
) -> ClickDecorator:
    option_kwargs: dict[str, Any] = {
        "type": click.Choice(tuple(choices)),
        "help": help,
    }
    if default is not None:
        option_kwargs["default"] = default
        option_kwargs["show_default"] = True
    else:
        option_kwargs["default"] = None
    return click.option("--device", **option_kwargs)


def dagzoo_root_option(*, help: str = "Local dagzoo checkout root") -> ClickDecorator:
    return path_option("dagzoo-root", required=True, help=help)


def sweep_id_option(*, help: str = "Sweep id to inspect") -> ClickDecorator:
    return click.option("--sweep-id", required=True, help=help)


def catalog_path_option(func: Any) -> Any:
    return path_option(
        "catalog-path",
        default=sweep_paths.default_catalog_path(),
        help="Path to reference/system_delta_catalog.yaml",
    )(func)


def index_path_option(func: Any) -> Any:
    return path_option(
        "index-path",
        default=sweep_paths.default_sweep_index_path(),
        help="Path to reference/system_delta_sweeps/index.yaml",
    )(func)


def sweeps_root_option(func: Any) -> Any:
    return path_option(
        "sweeps-root",
        default=sweep_paths.default_sweeps_root(),
        help="Path to reference/system_delta_sweeps/",
    )(func)


def sweep_registry_path_option(func: Any) -> Any:
    return path_option(
        "registry-path",
        default=sweep_paths.default_registry_path(),
        help="Path to benchmark_run_registry_v1.json",
    )(func)


def sweep_path_options(
    *,
    include_registry: bool,
    include_sweeps_root: bool,
) -> ClickDecorator:
    decorators: list[ClickDecorator] = [catalog_path_option, index_path_option]
    if include_sweeps_root:
        decorators.append(sweeps_root_option)
    if include_registry:
        decorators.append(sweep_registry_path_option)
    return apply_click_decorators(*decorators)


def materialize_worker_options(*, processes_help: str) -> ClickDecorator:
    return apply_click_decorators(
        click.option(
            "--materialize-processes",
            default=default_materialize_processes(),
            show_default=True,
            type=POSITIVE_INT,
            help=processes_help,
        ),
        click.option(
            "--materialize-worker-threads",
            default=None,
            type=POSITIVE_INT_OR_AUTO,
            help="Per-dagzoo subprocess CPU thread budget. Use 'auto' for the balanced default.",
        ),
    )


class PositiveIntType(click.ParamType):
    name = "positive-int"

    def convert(self, value: Any, param: click.Parameter | None, ctx: click.Context | None) -> int:
        try:
            converted = int(value)
        except (TypeError, ValueError):
            self.fail(f"Expected a positive integer, got {value}.", param, ctx)
        if converted <= 0:
            self.fail(f"Expected a positive integer, got {value}.", param, ctx)
        return converted


class PositiveIntOrAutoType(click.ParamType):
    name = "positive-int-or-auto"

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> int | None:
        if isinstance(value, str) and value.strip().lower() == "auto":
            return None
        return POSITIVE_INT.convert(value, param, ctx)


class UInt32Type(click.ParamType):
    name = "uint32"

    def convert(self, value: Any, param: click.Parameter | None, ctx: click.Context | None) -> int:
        try:
            converted = int(value)
        except (TypeError, ValueError):
            self.fail(f"Expected a 32-bit unsigned seed, got {value}.", param, ctx)
        if converted < 0 or converted > _UINT32_MAX:
            self.fail(f"Expected a 32-bit unsigned seed, got {value}.", param, ctx)
        return converted


class FiniteFloatType(click.ParamType):
    name = "finite-float"

    def __init__(self, *, flag_name: str) -> None:
        self.flag_name = flag_name

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> float:
        try:
            converted = float(value)
        except (TypeError, ValueError):
            self.fail(f"Invalid {self.flag_name} value {value!r}.", param, ctx)
        if converted != converted or converted in (float("inf"), float("-inf")):
            self.fail(f"Invalid {self.flag_name} value {value!r}.", param, ctx)
        return converted


class MissingRateType(FiniteFloatType):
    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> float:
        converted = super().convert(value, param, ctx)
        if converted < 0.0 or converted > 1.0:
            self.fail("Expected --missing-rate in [0, 1].", param, ctx)
        return converted


class MissingFractionType(FiniteFloatType):
    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> float:
        converted = super().convert(value, param, ctx)
        if converted <= 0.0 or converted > 1.0:
            self.fail("Expected --missing-mar-observed-fraction in (0, 1].", param, ctx)
        return converted


class PositiveFloatType(FiniteFloatType):
    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> float:
        converted = super().convert(value, param, ctx)
        if converted <= 0.0:
            self.fail(f"Expected {self.flag_name} > 0.", param, ctx)
        return converted


POSITIVE_INT = PositiveIntType()
POSITIVE_INT_OR_AUTO = PositiveIntOrAutoType()
UINT32 = UInt32Type()
