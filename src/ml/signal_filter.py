"""RandomForest trade gate (fail-open). Canonical Phase 8 ML path under ``ml/``."""

from __future__ import annotations

import argparse
import json
import logging
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from config.constants import LOGGER_STRATEGY
from config.settings import Settings, get_settings
from core.database import Database

_LOG = logging.getLogger(LOGGER_STRATEGY)

REGIME_ENCODE = {"Range": 0.0, "Trending": 1.0, "": 0.0}


def _symbol_encoding(symbol: str) -> float:
    raw = str(symbol or "UNK").strip().upper().encode("utf-8")
    return float(zlib.crc32(raw) & 0xFFFF) / 65535.0


FEATURE_NAMES = (
    "rsi",
    "adx",
    "sma_dist",
    "sentiment_score",
    "hour_utc",
    "dow_mon0",
    "atr",
    "spread_pct",
    "regime_num",
    "trailing_hint",
    "symbol_enc",
)


def _parse_ts(s: Any) -> datetime:
    raw = str(s or "").replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except ValueError:
        return datetime.now(UTC)


def build_feature_vector(
    md: dict[str, Any],
    row: dict[str, Any] | None = None,
    *,
    symbol: str = "",
) -> np.ndarray:
    """Features known at signal time (metadata + row sentiment/regime tops)."""

    rsi_v = float(md.get("rsi") or md.get("rsi_value") or 50.0)
    adx_v = float(md.get("adx") or 0.0)
    sma = float(md.get("sma200") or 0.0)
    last_c = float(md.get("last_close") or md.get("entry_price_anchor") or 0.0)
    sma_dist = (last_c - sma) / sma if abs(sma) > 1e-9 else 0.0
    sent = float((row or {}).get("sentiment_score") or md.get("sentiment_score") or 0.0)
    opened = _parse_ts(md.get("bar_timestamp") or (row or {}).get("opened_at"))
    hr = opened.hour + opened.minute / 60.0
    dow = float(opened.weekday())
    atr_v = float(md.get("atr") or 0.0)
    sp = float(md.get("spread_pct") or 0.0)
    reg = REGIME_ENCODE.get(str(md.get("regime_type") or (row or {}).get("regime_type") or "Range"), 0.0)
    trail_hint = 0.0
    sym = symbol or str(md.get("symbol") or (row or {}).get("symbol") or "")
    sym_enc = _symbol_encoding(sym)
    return np.array(
        [rsi_v, adx_v, sma_dist, sent, hr, dow, atr_v, sp, reg, trail_hint, sym_enc],
        dtype=np.float64,
    )


def _sklearn_rf():
    try:
        from sklearn.ensemble import RandomForestClassifier  # noqa: WPS433
        from sklearn.metrics import roc_auc_score  # noqa: WPS433
        from sklearn.model_selection import train_test_split  # noqa: WPS433

        return RandomForestClassifier, roc_auc_score, train_test_split
    except Exception:  # noqa: BLE001
        return None, None, None


def _maybe_joblib():
    try:
        import joblib  # noqa: WPS433

        return joblib
    except Exception:  # noqa: BLE001
        return None


@dataclass(frozen=True)
class MLDecision:
    allowed: bool
    probability: float | None
    reason: str
    model_trained: bool


class MLSignalFilter:
    """RandomForest classifier; defaults to fail-open."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model_path = Path(settings.ML_MODEL_PATH)
        self._meta_path = Path(settings.ML_MODEL_META_PATH)
        self._model: Any = None
        self._block_entries_due_to_startup_failure = False
        self._load_model()

    def _load_model(self) -> None:
        jl = _maybe_joblib()
        if jl is None:
            self._model = None
            return
        mp = self._model_path if self._model_path.is_absolute() else Path.cwd() / self._model_path
        if mp.is_file():
            try:
                self._model = jl.load(mp)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("event=ml_filter_train_skipped cannot_load_model err=%s", exc)
                self._model = None

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def mark_startup_training_failure(self) -> None:
        """Fail-closed gate: block entries after startup training exception."""

        self._block_entries_due_to_startup_failure = True

    def clear_startup_training_block(self) -> None:
        self._block_entries_due_to_startup_failure = False

    def train_from_database(self, db: Database) -> None:
        if not bool(self._settings.ENABLE_ML_FILTER):
            _LOG.info("event=ml_filter_train_skipped reason=disabled")
            return

        jl = _maybe_joblib()
        rf_classifier, roc_auc, ttsplit = _sklearn_rf()
        if jl is None or rf_classifier is None:
            _LOG.warning("event=ml_filter_train_skipped reason=missing_sklearn_joblib")
            return

        rows = db.get_ml_training_rows(
            limit=int(self._settings.ML_MAX_TRAINING_TRADES),
            exclude_simulation=True,
        )
        recent_n = min(len(rows), int(self._settings.ML_INFERENCE_RECENT_CONTEXT))

        min_n = int(self._settings.MIN_ML_TRAINING_TRADES)
        if len(rows) < min_n:
            _LOG.info(
                "event=ml_filter_train_skipped reason=insufficient_trades have=%s need=%s",
                len(rows),
                min_n,
            )
            return

        xs: list[list[float]] = []
        ys: list[int] = []
        rows_sorted = sorted(
            rows,
            key=lambda r: str(r.get("closed_at") or r.get("opened_at") or ""),
        )
        span = min(len(rows_sorted), int(self._settings.ML_MAX_TRAINING_TRADES))
        used = rows_sorted[-span:]
        for r in used:
            meta = r.get("metadata") or {}
            if not isinstance(meta, dict):
                continue
            y = 1 if float(r["realized_pnl"]) > 1e-9 else 0
            try:
                sym = str(r.get("symbol") or "")
                vec = build_feature_vector(meta, r, symbol=sym)
                xs.append(vec.tolist())
                ys.append(y)
            except Exception:  # noqa: BLE001
                continue

        if len(xs) < min_n:
            _LOG.info("event=ml_filter_train_skipped reason=could_not_build_features")
            return

        _LOG.info("event=ml_filter_train_start rows=%s recent_context_window=%s", len(xs), recent_n)

        feature_matrix = np.array(xs, dtype=np.float64)
        y_labels = np.array(ys, dtype=np.int64)
        clf = rf_classifier(
            n_estimators=120,
            max_depth=6,
            random_state=42,
            min_samples_leaf=3,
        )
        acc = auc = None
        if len(y_labels) >= 30 and roc_auc is not None and ttsplit is not None:
            try:
                xa, xv, ya, yv = ttsplit(feature_matrix, y_labels, test_size=0.25, shuffle=False)
                clf.fit(xa, ya)
                pred = clf.predict(xv)
                acc = float((pred == yv).mean())
                if len(set(yv.tolist())) > 1:
                    prob = clf.predict_proba(xv)[:, 1]
                    auc = float(roc_auc(yv, prob))
                else:
                    auc = None
                clf.fit(feature_matrix, y_labels)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("event=ml_filter_train_partial_validation err=%s", exc)
                clf.fit(feature_matrix, y_labels)
                acc = auc = None
        else:
            clf.fit(feature_matrix, y_labels)

        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        jl.dump(clf, self._model_path if self._model_path.is_absolute() else Path.cwd() / self._model_path)
        self._model = clf
        self.clear_startup_training_block()
        trained_at = datetime.now(UTC).isoformat()
        meta_payload = {
            "trained_at": trained_at,
            "training_trade_count": int(len(xs)),
            "recent_context_trades": int(recent_n),
            "features": list(FEATURE_NAMES),
            "validation_accuracy": acc,
            "validation_auc": auc,
            "source_database": str(db.path),
            "threshold": float(self._settings.ML_FILTER_THRESHOLD),
        }
        mp = self._meta_path if self._meta_path.is_absolute() else Path.cwd() / self._meta_path
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
        _LOG.info(
            "event=ml_filter_train_complete trades=%s acc=%s auc=%s threshold=%s",
            len(xs),
            acc if acc is not None else "n_a",
            auc if auc is not None else "n_a",
            self._settings.ML_FILTER_THRESHOLD,
        )

    def should_allow_trade(self, *, signal_context: dict[str, Any]) -> MLDecision:
        """``signal_context`` should include at least ``symbol`` and entry-time indicator fields."""

        symbol = str(signal_context.get("symbol") or "")
        thr = float(self._settings.ML_FILTER_THRESHOLD)
        if not self._settings.ENABLE_ML_FILTER:
            return MLDecision(True, None, "disabled", False)

        if self._block_entries_due_to_startup_failure:
            _LOG.warning(
                "event=ml_inference_blocked symbol=%s reason=startup_training_failed",
                symbol,
                extra={"symbol": symbol},
            )
            return MLDecision(False, None, "startup_training_failed", False)

        if self._model is None:
            _LOG.info(
                "event=ml_filter_fail_open symbol=%s reason=no_model",
                symbol,
                extra={"symbol": symbol},
            )
            return MLDecision(True, None, "no_model", False)

        try:
            md = {k: v for k, v in signal_context.items() if k != "symbol"}
            vec = build_feature_vector(md, symbol=symbol).reshape(1, -1)
            proba = float(self._model.predict_proba(vec)[0][1])
        except Exception as exc:  # noqa: BLE001
            _LOG.info(
                "event=ml_filter_fail_open symbol=%s reason=infer_error err=%s",
                symbol,
                exc,
                extra={"symbol": symbol},
            )
            _LOG.info(
                "event=ml_filter_inference symbol=%s probability=n_a threshold=%.6f decision=fail_open reason=%s",
                symbol,
                thr,
                exc,
                extra={"symbol": symbol},
            )
            return MLDecision(True, None, f"infer_error:{exc}", self.is_trained)

        ok = bool(proba >= thr)
        decision = "allow" if ok else "block"
        feat_hint = "|".join(f"{FEATURE_NAMES[j]}:{float(vec[0, j]):.4g}" for j in range(vec.shape[1]))
        _LOG.info(
            "event=strategy_signal symbol=%s ml_filter_enabled=true ml_model_trained=true "
            "ml_probability=%.6f ml_threshold=%.6f ml_decision=%s ml_reason=ml_classifier features=%s",
            symbol,
            proba,
            thr,
            decision,
            feat_hint[:420],
            extra={"symbol": symbol},
        )
        _LOG.info(
            "event=ml_filter_inference symbol=%s probability=%.6f threshold=%.6f decision=%s reason=inferred",
            symbol,
            proba,
            thr,
            decision,
            extra={"symbol": symbol},
        )

        if not ok:
            _LOG.info(
                "event=ml_trade_blocked symbol=%s probability=%.6f threshold=%.6f",
                symbol,
                proba,
                thr,
                extra={"symbol": symbol},
            )

        return MLDecision(ok, proba, "below_threshold" if not ok else "inferred", True)

    def predict_gate(self, *, entry_metadata: dict[str, Any], symbol: str) -> MLDecision:
        """Backward-compatible alias."""

        ctx = dict(entry_metadata)
        ctx["symbol"] = symbol
        return self.should_allow_trade(signal_context=ctx)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Train ML signal filter from SQLite trades.")
    p.add_argument("--train", action="store_true")
    args = p.parse_args()
    if not args.train:
        p.error("provide --train")
    settings = get_settings()
    dbp = Path(settings.DATABASE_PATH)
    resolved = dbp if dbp.is_absolute() else Path.cwd() / dbp
    db = Database(resolved)
    MLSignalFilter(settings).train_from_database(db)


if __name__ == "__main__":
    main()


__all__ = [
    "FEATURE_NAMES",
    "MLDecision",
    "MLSignalFilter",
    "build_feature_vector",
]
