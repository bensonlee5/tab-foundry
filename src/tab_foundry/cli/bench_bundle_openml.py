"""CLI wiring for `tab-foundry bench bundle build-openml`."""

from __future__ import annotations

from pathlib import Path
import sys

import click

import tab_foundry.bench.openml_bundle as bundle_module
from tab_foundry.cli.click_utils import POSITIVE_INT, run_click_command


def _build_openml_bundle_command(
    *,
    out_path: Path,
    bundle_name: str,
    version: int,
    task_source: str,
    discover_from_openml: bool,
    new_instances: int,
    min_instances: int,
    min_task_count: int,
    task_type: str,
    max_features: int,
    min_classes: int | None,
    max_classes: str,
    min_missing_pct: float,
    max_missing_pct: float,
    min_minority_class_pct: float,
) -> int:
    config = bundle_module.OpenMLBenchmarkBundleConfig(
        bundle_name=bundle_name,
        version=version,
        task_source=task_source,
        task_type=task_type,
        new_instances=new_instances,
        max_features=max_features,
        min_classes=min_classes,
        max_classes=bundle_module.parse_max_classes_arg(max_classes),
        min_missing_pct=min_missing_pct,
        max_missing_pct=max_missing_pct,
        min_minority_class_pct=min_minority_class_pct,
        discover_from_openml=discover_from_openml,
        min_instances=min_instances,
        min_task_count=min_task_count,
    )
    if config.discover_from_openml:
        build_result = bundle_module.build_openml_benchmark_bundle_result(config)
        report = bundle_module.render_openml_benchmark_candidate_report(build_result.report_entries)
        if report:
            print(report)
        out_path = bundle_module.write_openml_benchmark_bundle(
            out_path,
            config,
            bundle=build_result.bundle,
        )
    else:
        out_path = bundle_module.write_openml_benchmark_bundle(out_path, config)
    print(f"wrote benchmark bundle: {out_path}")
    return 0


@click.command(name="build-openml", help="Build an OpenML benchmark bundle")
@click.option("--out-path", required=True, type=click.Path(path_type=Path), help="JSON output path for the bundle")
@click.option("--bundle-name", required=True, help="Bundle name persisted in the JSON payload")
@click.option("--version", required=True, type=int, help="Bundle version persisted in the JSON payload")
@click.option(
    "--task-source",
    default=bundle_module.DEFAULT_OPENML_TASK_SOURCE,
    show_default=True,
    type=click.Choice(bundle_module.task_source_names()),
    help="Named pinned OpenML task-id source pool used before applying bundle filters",
)
@click.option(
    "--discover-from-openml",
    is_flag=True,
    help="Query OpenML task metadata directly instead of using a pinned task-source registry",
)
@click.option("--new-instances", default=200, show_default=True, type=POSITIVE_INT, help="Subsampled row count used by the benchmark")
@click.option(
    "--min-instances",
    default=1,
    show_default=True,
    type=POSITIVE_INT,
    help="Minimum raw dataset row count required during OpenML discovery",
)
@click.option(
    "--min-task-count",
    default=1,
    show_default=True,
    type=POSITIVE_INT,
    help="Minimum validated task count required after OpenML discovery",
)
@click.option(
    "--task-type",
    default="supervised_classification",
    show_default=True,
    type=click.Choice(("supervised_classification", "supervised_regression")),
    help="OpenML task type used when building the benchmark bundle",
)
@click.option("--max-features", default=10, show_default=True, type=POSITIVE_INT, help="Maximum raw OpenML feature count allowed by the bundle filter")
@click.option(
    "--min-classes",
    default=None,
    type=POSITIVE_INT,
    help="Minimum class count filter for classification bundles",
)
@click.option(
    "--max-classes",
    default="2",
    show_default=True,
    help="Maximum class count filter, or 'auto' to widen to the highest eligible class count",
)
@click.option(
    "--min-missing-pct",
    default=0.0,
    show_default=True,
    type=float,
    help="Minimum required percentage of instances with missing values",
)
@click.option(
    "--max-missing-pct",
    default=0.0,
    show_default=True,
    type=float,
    help="Maximum allowed percentage of instances with missing values",
)
@click.option(
    "--min-minority-class-pct",
    default=2.5,
    show_default=True,
    type=float,
    help="Minimum required minority class percentage",
)
def COMMAND(
    out_path: Path,
    bundle_name: str,
    version: int,
    task_source: str,
    discover_from_openml: bool,
    new_instances: int,
    min_instances: int,
    min_task_count: int,
    task_type: str,
    max_features: int,
    min_classes: int | None,
    max_classes: str,
    min_missing_pct: float,
    max_missing_pct: float,
    min_minority_class_pct: float,
) -> int:
    return _build_openml_bundle_command(
        out_path=out_path,
        bundle_name=bundle_name,
        version=version,
        task_source=task_source,
        discover_from_openml=discover_from_openml,
        new_instances=new_instances,
        min_instances=min_instances,
        min_task_count=min_task_count,
        task_type=task_type,
        max_features=max_features,
        min_classes=min_classes,
        max_classes=max_classes,
        min_missing_pct=min_missing_pct,
        max_missing_pct=max_missing_pct,
        min_minority_class_pct=min_minority_class_pct,
    )


def main(argv: list[str] | None = None) -> int:
    return run_click_command(COMMAND, argv, prog_name="tab-foundry bench bundle build-openml")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
