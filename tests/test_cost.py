import dataclasses
import math

from grblr.cost import AverageProxyJoinCost, TargetCost, compute_norm_stats
from grblr.run_select import InstanceFeatures


def _feat(**kw):
    base = InstanceFeatures(
        word_id=1,
        duration_ms=200,
        rms_db=-20.0,
        f0=120.0,
        centroid=1500.0,
        voiced=0.8,
        quality=None,
        keep_prob=None,
    )
    return dataclasses.replace(base, **kw)


def test_norm_stats_mean_std():
    feats = {(1, 0): _feat(rms_db=-10.0), (1, 1): _feat(rms_db=-30.0)}
    ns = compute_norm_stats(feats)
    assert abs(ns.mean["rms_db"] + 20.0) < 1e-9
    assert ns.std["rms_db"] > 0


def test_target_cost_prefers_higher_quality():
    feats = {
        (1, 0): _feat(keep_prob=0.9, quality="good"),
        (1, 1): _feat(keep_prob=0.2, quality="ignore"),
    }
    ns = compute_norm_stats(feats)
    tc = TargetCost(ns)
    good = tc.run_cost([(1, 0)], feats)
    bad = tc.run_cost([(1, 1)], feats)
    assert good < bad


def test_target_cost_absent_keep_prob_drops_cleanly():
    feats = {(1, 0): _feat(keep_prob=None)}
    ns = compute_norm_stats(feats)
    c = TargetCost(ns).run_cost([(1, 0)], feats)
    assert math.isfinite(c)


def test_join_cost_lower_for_continuous_seam():
    feats = {
        (1, 0): _feat(f0=120.0, rms_db=-20.0, centroid=1500.0),  # A.end
        (2, 0): _feat(f0=122.0, rms_db=-20.5, centroid=1510.0),  # B.start (close)
        (3, 0): _feat(f0=300.0, rms_db=-5.0, centroid=4000.0),  # B'.start (far)
    }
    ns = compute_norm_stats(feats)
    jc = AverageProxyJoinCost(ns)
    smooth = jc.cost((1, 0), (2, 0), feats)
    rough = jc.cost((1, 0), (3, 0), feats)
    assert smooth < rough


def test_join_cost_handles_null_f0():
    feats = {(1, 0): _feat(f0=None), (2, 0): _feat(f0=None)}
    ns = compute_norm_stats(feats)
    import math

    assert math.isfinite(AverageProxyJoinCost(ns).cost((1, 0), (2, 0), feats))


def test_join_cost_fallback_node_is_neutral():
    feats = {(1, 0): _feat()}
    ns = compute_norm_stats(feats)
    jc = AverageProxyJoinCost(ns)
    import math

    assert jc.cost(None, (1, 0), feats) == jc.NEUTRAL
    assert math.isfinite(jc.cost((1, 0), None, feats))
