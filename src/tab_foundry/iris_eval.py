"""Iris evaluation helpers for tab-foundry checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from tab_foundry.checkpoint_classifier import TabFoundryClassifier


@dataclass(slots=True)
class IrisEvalSummary:
    """Aggregate Iris benchmark results."""

    checkpoint: Path
    results: dict[str, list[float]]


def evaluate_iris_checkpoint(
    checkpoint_path: Path,
    *,
    device: str = "cpu",
    seeds: int = 5,
) -> IrisEvalSummary:
    """Evaluate a classification checkpoint on binary Iris splits."""

    classifier = TabFoundryClassifier(checkpoint_path, device=device)

    iris = load_iris()
    x = iris.data[iris.target != 0]
    y = iris.target[iris.target != 0] - 1

    results: dict[str, list[float]] = {
        "tab_foundry": [],
        "LogReg": [],
        "RF": [],
        "KNN": [],
        "Tree": [],
    }
    for seed in range(seeds):
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.5,
            random_state=seed,
        )
        classifier.fit(x_train, y_train)
        results["tab_foundry"].append(float(roc_auc_score(y_test, classifier.predict_proba(x_test)[:, 1])))
        results["LogReg"].append(
            float(
                roc_auc_score(
                    y_test,
                    LogisticRegression(max_iter=1000).fit(x_train, y_train).predict_proba(x_test)[:, 1],
                )
            )
        )
        results["RF"].append(
            float(
                roc_auc_score(
                    y_test,
                    RandomForestClassifier(random_state=seed).fit(x_train, y_train).predict_proba(
                        x_test
                    )[:, 1],
                )
            )
        )
        results["KNN"].append(
            float(
                roc_auc_score(
                    y_test,
                    KNeighborsClassifier().fit(x_train, y_train).predict_proba(x_test)[:, 1],
                )
            )
        )
        results["Tree"].append(
            float(
                roc_auc_score(
                    y_test,
                    DecisionTreeClassifier(random_state=seed).fit(x_train, y_train).predict_proba(
                        x_test
                    )[:, 1],
                )
            )
        )
    return IrisEvalSummary(checkpoint=checkpoint_path.expanduser().resolve(), results=results)
