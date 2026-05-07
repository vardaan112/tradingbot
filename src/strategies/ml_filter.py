"""Backward-compatible import path — canonical implementation lives in ``ml.signal_filter``."""

from __future__ import annotations

from ml.signal_filter import FEATURE_NAMES, MLDecision, MLSignalFilter, build_feature_vector

__all__ = ["FEATURE_NAMES", "MLDecision", "MLSignalFilter", "build_feature_vector"]
