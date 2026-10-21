"""H&B-faithful cost models for the run-selection lattice (spec 2026-06-01).
Target cost = std-normalized unit-quality prior (the reduction of H&B C^t when
there is no prosodic target spec). Join cost is a pluggable protocol; the v1
AverageProxyJoinCost scores seams between the two real clips' per-word average
features. Weights are equal across std-normalized sub-costs (faithful bootstrap);
no per-render tuning."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Protocol

from grblr.run_select import InstanceFeatures

Pos = tuple[int, int]  # (clip_id, position)

# z-normalized scalar features used by target cost (quality prior).
_TARGET_NUM = ("duration_ms", "rms_db", "voiced", "keep_prob")
# features whose z-discontinuity at a seam forms the join cost.
_JOIN_NUM = ("f0", "rms_db", "centroid")


@dataclass(frozen=True)
class NormStats:
    mean: dict[str, float]
    std: dict[str, float]


def _values(feats: dict[Pos, InstanceFeatures], attr: str) -> list[float]:
    out = []
    for f in feats.values():
        v = getattr(f, attr)
        if v is not None:
            out.append(float(v))
    return out


def compute_norm_stats(feats: dict[Pos, InstanceFeatures]) -> NormStats:
    """Corpus mean/std per feature over the filtered index (one pass). std falls
    back to 1.0 for constant/empty features so z-scores stay finite."""
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for attr in set(_TARGET_NUM) | set(_JOIN_NUM):
        vals = _values(feats, attr)
        if len(vals) >= 2:
            mean[attr] = statistics.fmean(vals)
            s = statistics.pstdev(vals)
            std[attr] = s if s > 1e-9 else 1.0
        else:
            mean[attr] = vals[0] if vals else 0.0
            std[attr] = 1.0
    return NormStats(mean=mean, std=std)


def _z(ns: NormStats, attr: str, v: float | None) -> float | None:
    if v is None:
        return None
    return (float(v) - ns.mean[attr]) / ns.std[attr]


class TargetCost:
    """Per-instance quality prior. Lower cost = better unit. A run's cost is the
    mean of its words' per-word costs (so long runs aren't penalized for length)."""

    def __init__(self, ns: NormStats):
        self.ns = ns

    def _word_cost(self, f: InstanceFeatures) -> float:
        terms: list[float] = []
        # Higher keep_prob / voiced / duration / loudness => lower cost (negate z).
        for attr in _TARGET_NUM:
            z = _z(self.ns, attr, getattr(f, attr))
            if z is not None:
                terms.append(-z)
        # Quality label: 'good' rewarded, 'ignore' penalized (others neutral).
        if f.quality == "good":
            terms.append(-1.0)
        elif f.quality == "ignore":
            terms.append(1.0)
        return statistics.fmean(terms) if terms else 0.0

    def run_cost(
        self, positions: list[Pos], feats: dict[Pos, InstanceFeatures]
    ) -> float:
        costs = [self._word_cost(feats[p]) for p in positions]
        return statistics.fmean(costs) if costs else 0.0


class JoinCostModel(Protocol):
    def cost(
        self, a: Pos | None, b: Pos | None, feats: dict[Pos, InstanceFeatures]
    ) -> float: ...


class AverageProxyJoinCost:
    """v1 join cost: std-normalized mean |z_A - z_B| over F0/power/centroid, using
    the two real clips' per-word AVERAGE features as edge-frame proxies (spec
    §4.2 -- the instance is known; only the feature fidelity is crude). A seam
    touching a fallback/MISS node (None) gets a neutral constant."""

    NEUTRAL = 1.0

    def __init__(self, ns: NormStats):
        self.ns = ns

    def cost(
        self, a: Pos | None, b: Pos | None, feats: dict[Pos, InstanceFeatures]
    ) -> float:
        if a is None or b is None:
            return self.NEUTRAL
        fa, fb = feats[a], feats[b]
        diffs: list[float] = []
        for attr in _JOIN_NUM:
            za = _z(self.ns, attr, getattr(fa, attr))
            zb = _z(self.ns, attr, getattr(fb, attr))
            if za is not None and zb is not None:
                diffs.append(abs(za - zb))
        return statistics.fmean(diffs) if diffs else self.NEUTRAL
