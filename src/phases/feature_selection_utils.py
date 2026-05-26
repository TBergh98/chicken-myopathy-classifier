#!/usr/bin/env python3
"""Shared helpers for feature-selection artifacts and model inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def write_feature_manifest(
    path: Path,
    *,
    name: str,
    source_phase: str,
    selection_rule: str,
    features: Iterable[str],
    metadata: dict | None = None,
) -> None:
    """Write a small machine-readable feature manifest."""

    feature_list = list(dict.fromkeys(features))
    payload = {
        "name": name,
        "source_phase": source_phase,
        "selection_rule": selection_rule,
        "n_features": len(feature_list),
        "features": feature_list,
    }
    if metadata:
        payload["metadata"] = metadata

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_feature_manifest(path: Path) -> dict:
    """Load a feature manifest from disk."""

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if "features" not in payload or not isinstance(payload["features"], list):
        raise ValueError(f"Invalid feature manifest: {path}")
    return payload


def deduplicate_features(features: Iterable[str]) -> list[str]:
    """Preserve order while removing duplicates."""

    return list(dict.fromkeys(features))