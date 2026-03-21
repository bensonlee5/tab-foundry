from __future__ import annotations

import functools
import inspect

from . import exporter_cases as cases


def _export(names: list[str]) -> None:
    for name in names:
        case = getattr(cases, name)

        def _wrapped(*args, __case=case, **kwargs):
            return __case(*args, **kwargs)

        functools.update_wrapper(_wrapped, case)
        _wrapped.__name__ = name
        _wrapped.__qualname__ = name
        _wrapped.__signature__ = inspect.signature(case)
        globals()[name] = _wrapped


_export(
    [
        "test_model_config_round_trip_across_eval_export_and_loader",
        "test_reference_consumer_classification_matches_golden_fixture",
        "test_reference_consumer_rejects_v2_bundle",
        "test_reference_consumer_derives_preprocessing_from_runtime_support_set",
        "test_reference_consumer_rejects_missing_inputs_for_embedded_no_impute_policy",
        "test_reference_consumer_executes_embedded_no_impute_policy_on_finite_inputs",
        "test_reference_consumer_applies_embedded_nondefault_all_nan_fill",
        "test_reference_consumer_rejects_nonfinite_class_probabilities",
        "test_reference_batch_rejects_non_classification_before_preprocessing",
        "test_reference_consumer_rejects_underwidth_logits",
    ]
)
