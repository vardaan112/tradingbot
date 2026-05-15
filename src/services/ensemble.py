"""Weighted ensemble decision engine (Phase 6 static + Phase 8 performance weights)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from strategies.base import Signal, SignalAction
from strategies.registry import normalize_strategy_name

if TYPE_CHECKING:
    from config.settings import Settings
    from core.database import Database


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


@dataclass
class StrategyVote:
    strategy_name: str
    symbol: str
    action: SignalAction
    confidence: float
    weight: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
    reference_price: float = 0.0


@dataclass
class EnsembleDecision:
    symbol: str
    final_action: SignalAction
    weighted_enter_score: float
    weighted_exit_score: float
    enter_threshold: float
    exit_threshold: float
    contributing_votes: list[StrategyVote]
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


def votes_to_contributing_json(votes: list[StrategyVote]) -> str:
    payload = [
        {
            "strategy_name": v.strategy_name,
            "symbol": v.symbol,
            "action": v.action.value,
            "confidence": v.confidence,
            "weight": v.weight,
            "reason": v.reason,
            "reference_price": v.reference_price,
            "metadata": v.metadata,
        }
        for v in votes
    ]
    return json.dumps(payload, separators=(",", ":"))


class WeightedEnsembleEngine:
    """Aggregate per-symbol ``Signal`` list into a single ensemble action."""

    def __init__(
        self,
        settings: "Settings",
        database: Optional["Database"] = None,
    ) -> None:
        self._settings = settings
        self._database = database
        self._prev_smoothed_perf: Optional[dict[str, float]] = None
        self._weights = self._build_normalized_weights(record_decision=False)

    def _build_static_weights(self) -> dict[str, float]:
        active = self._settings.active_strategies_list
        raw = self._settings.strategy_weights_dict
        lo = float(self._settings.ENSEMBLE_MIN_WEIGHT)
        hi = float(self._settings.ENSEMBLE_MAX_WEIGHT)
        ws: dict[str, float] = {}
        for name in active:
            key = normalize_strategy_name(name)
            w = float(raw.get(key, 1.0))
            w = max(lo, min(hi, w))
            ws[key] = w
        s = sum(ws.values())
        if s <= 0:
            n = max(1, len(active))
            return {normalize_strategy_name(x): 1.0 / n for x in active}
        return {k: v / s for k, v in ws.items()}

    def _build_normalized_weights(self, *, record_decision: bool) -> dict[str, float]:
        if self._settings.ENSEMBLE_WEIGHT_MODE != "performance":
            return self._build_static_weights()
        if self._database is None:
            return self._build_static_weights()

        from services.performance_weights import PerformanceWeightCalculator

        calc = PerformanceWeightCalculator(self._settings, self._database)
        result = calc.compute(
            previous_smoothed_weights=self._prev_smoothed_perf,
            record=record_decision,
        )
        if not result.used_performance or result.fallback_reason:
            self._prev_smoothed_perf = None
            return self._build_static_weights()
        self._prev_smoothed_perf = dict(result.weights)
        return dict(result.weights)

    def refresh_weights(self, *, record_decision: bool = True) -> None:
        """Rebuild weights (call periodically in live; optional each bar in replay)."""

        self._weights = self._build_normalized_weights(record_decision=record_decision)

    @property
    def effective_weights(self) -> dict[str, float]:
        """Current normalized ensemble weights (read-only)."""

        return dict(self._weights)

    @staticmethod
    def _latest_signal_per_strategy(signals: list[Signal], symbol: str) -> dict[str, Signal]:
        sym_u = symbol.strip().upper()
        out: dict[str, Signal] = {}
        for s in signals:
            if s.symbol.strip().upper() != sym_u:
                continue
            raw_name = str(s.strategy_name or "").strip()
            if not raw_name:
                continue
            key = normalize_strategy_name(raw_name)
            out[key] = s
        return out

    def decide(
        self,
        symbol: str,
        signals: list[Signal],
        *,
        has_position: bool,
    ) -> EnsembleDecision:
        sym_u = symbol.strip().upper()
        active = [normalize_strategy_name(n) for n in self._settings.active_strategies_list]
        by_strat = self._latest_signal_per_strategy(signals, sym_u)

        votes: list[StrategyVote] = []
        for name in active:
            w = float(self._weights.get(name, 1.0 / max(1, len(active))))
            sig = by_strat.get(name)
            if sig is None:
                votes.append(
                    StrategyVote(
                        strategy_name=name,
                        symbol=sym_u,
                        action=SignalAction.NONE,
                        confidence=0.0,
                        weight=w,
                        reason="no_signal_this_tick",
                        metadata={},
                        reference_price=0.0,
                    ),
                )
                continue
            conf = _clamp01(float(sig.confidence))
            votes.append(
                StrategyVote(
                    strategy_name=name,
                    symbol=sym_u,
                    action=sig.action,
                    confidence=conf,
                    weight=w,
                    reason=sig.reason,
                    metadata=dict(sig.metadata) if sig.metadata else {},
                    reference_price=float(sig.reference_price or 0.0),
                ),
            )

        enter_thr = float(self._settings.ENSEMBLE_ENTER_THRESHOLD)
        exit_thr = float(self._settings.ENSEMBLE_EXIT_THRESHOLD)
        min_agree = max(1, int(self._settings.ENSEMBLE_MIN_AGREEING_STRATEGIES))
        policy = str(self._settings.ENSEMBLE_EXIT_POLICY).strip().lower()

        emergency = [v for v in votes if v.action == SignalAction.EMERGENCY_EXIT_LONG]
        if emergency:
            return EnsembleDecision(
                symbol=sym_u,
                final_action=SignalAction.EMERGENCY_EXIT_LONG,
                weighted_enter_score=0.0,
                weighted_exit_score=1.0,
                enter_threshold=enter_thr,
                exit_threshold=exit_thr,
                contributing_votes=list(votes),
                reason="emergency_exit_override",
                metadata={"policy": policy, "emergency_strategies": [e.strategy_name for e in emergency]},
            )

        exit_score = sum(v.weight * v.confidence for v in votes if v.action == SignalAction.EXIT_LONG)
        n_exit = sum(1 for v in votes if v.action == SignalAction.EXIT_LONG and v.confidence > 1e-12)

        want_exit = False
        if has_position:
            if policy == "any":
                want_exit = n_exit >= 1
            else:
                want_exit = bool(exit_score >= exit_thr and n_exit >= min_agree)

        enter_score = sum(v.weight * v.confidence for v in votes if v.action == SignalAction.ENTER_LONG)
        n_enter = sum(1 for v in votes if v.action == SignalAction.ENTER_LONG and v.confidence > 1e-12)

        want_enter = False
        if not has_position:
            want_enter = bool(enter_score >= enter_thr and n_enter >= min_agree)

        meta = {
            "enter_score": enter_score,
            "exit_score": exit_score,
            "n_enter": n_enter,
            "n_exit": n_exit,
            "policy": policy,
            "has_position": has_position,
        }

        if want_exit:
            return EnsembleDecision(
                symbol=sym_u,
                final_action=SignalAction.EXIT_LONG,
                weighted_enter_score=enter_score,
                weighted_exit_score=exit_score,
                enter_threshold=enter_thr,
                exit_threshold=exit_thr,
                contributing_votes=list(votes),
                reason="ensemble_exit",
                metadata=meta,
            )
        if want_enter:
            return EnsembleDecision(
                symbol=sym_u,
                final_action=SignalAction.ENTER_LONG,
                weighted_enter_score=enter_score,
                weighted_exit_score=exit_score,
                enter_threshold=enter_thr,
                exit_threshold=exit_thr,
                contributing_votes=list(votes),
                reason="ensemble_enter",
                metadata=meta,
            )

        return EnsembleDecision(
            symbol=sym_u,
            final_action=SignalAction.NONE,
            weighted_enter_score=enter_score,
            weighted_exit_score=exit_score,
            enter_threshold=enter_thr,
            exit_threshold=exit_thr,
            contributing_votes=list(votes),
            reason="ensemble_none",
            metadata=meta,
        )

    def to_signal(self, decision: EnsembleDecision) -> Signal:
        """Map ``EnsembleDecision`` to a ``Signal`` for the existing risk / order pipeline."""

        ref = 0.0
        if decision.final_action == SignalAction.ENTER_LONG:
            num = den = 0.0
            for v in decision.contributing_votes:
                if v.action == SignalAction.ENTER_LONG and v.confidence > 0 and v.reference_price > 0:
                    num += v.reference_price * v.weight * v.confidence
                    den += v.weight * v.confidence
            ref = num / den if den > 1e-18 else 0.0
        elif decision.final_action in (SignalAction.EXIT_LONG, SignalAction.EMERGENCY_EXIT_LONG):
            num = den = 0.0
            for v in decision.contributing_votes:
                if v.action in (SignalAction.EXIT_LONG, SignalAction.EMERGENCY_EXIT_LONG) and v.confidence > 0:
                    px = v.reference_price if v.reference_price > 0 else 0.0
                    if px > 0:
                        num += px * v.weight * v.confidence
                        den += v.weight * v.confidence
            ref = num / den if den > 1e-18 else 0.0

        return Signal(
            symbol=decision.symbol,
            action=decision.final_action,
            reason=decision.reason,
            reference_price=float(ref),
            atr=0.0,
            metadata={
                "ensemble": True,
                "weighted_enter_score": decision.weighted_enter_score,
                "weighted_exit_score": decision.weighted_exit_score,
                **decision.metadata,
            },
            strategy_name="ensemble",
            confidence=1.0,
        )


__all__ = [
    "EnsembleDecision",
    "StrategyVote",
    "WeightedEnsembleEngine",
    "votes_to_contributing_json",
]
