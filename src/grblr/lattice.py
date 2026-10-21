"""Instance-level run-selection lattice + pruned Viterbi (spec 2026-06-01).
Nodes are (span[i:j], specific clip instance); edges carry join cost between the
two real clips at a seam. A length-1 FALLBACK node at every position guarantees a
connected DAG so a zero-instance literal word degrades to the per-word path."""

from __future__ import annotations

from dataclasses import dataclass

from grblr.run_select import PositionIndex, all_run_candidates

FALLBACK = object()  # sentinel instance for zero-instance literal words

Pos = tuple[int, int]


@dataclass(eq=False)  # identity hash/eq so Node can key the Viterbi memo
class Node:
    i: int  # start position (inclusive) in the target
    j: int  # end position (exclusive)
    clip_id: int | None  # None for a fallback node
    first_pos: Pos | None
    last_pos: Pos | None
    start_ms: int | None
    end_ms: int | None
    instance: object  # FALLBACK, or the chosen clip (clip_id) -- concrete unit
    target_cost: float
    words: list[str]
    n_candidates: int = 0  # # of clip instances that could fill this span (pre-prune)


@dataclass
class Lattice:
    nodes: list[Node]
    n: int  # target length


def build_lattice(
    idx: PositionIndex, target: list[str], target_cost, top_k: int = 16
) -> Lattice:
    assert top_k >= 1, "build_lattice requires top_k >= 1 (else a span emits no node)"
    nodes: list[Node] = []
    fallback_cost = target_cost.run_cost([], idx.feat)  # constant (no positions)
    for i in range(len(target)):
        cands = all_run_candidates(idx, target, i)  # (length, clip_id, first_pos)
        scored = []
        for length, clip_id, pos in cands:
            positions = [(clip_id, pos + k) for k in range(length)]
            tcost = target_cost.run_cost(positions, idx.feat)
            scored.append((tcost, length, clip_id, pos, positions))
        # group by span length j; keep top_k instances per (i, j) by target cost
        by_len: dict[int, list] = {}
        for s in scored:
            by_len.setdefault(s[1], []).append(s)
        for length, group in by_len.items():
            group.sort(key=lambda s: (s[0], s[2], s[3]))  # cost, clip_id, pos
            n_cand = len(group)  # instances competing for this span, before top_k prune
            for tcost, _, clip_id, _pos, positions in group[:top_k]:
                first_pos, last_pos = positions[0], positions[-1]
                nodes.append(
                    Node(
                        i=i,
                        j=i + length,
                        clip_id=clip_id,
                        first_pos=first_pos,
                        last_pos=last_pos,
                        start_ms=idx.by_pos[first_pos][1],
                        end_ms=idx.by_pos[last_pos][2],
                        instance=clip_id,
                        target_cost=tcost,
                        words=target[i : i + length],
                        n_candidates=n_cand,
                    )
                )
        # connectivity guarantee: a length-1 fallback when the word has no instance.
        # `1 in by_len` iff a real length-1 node was emitted at this position (every
        # matching clip yields a length-1 candidate), so this is the cheap O(1)
        # equivalent of scanning the whole nodes list.
        if 1 not in by_len:
            nodes.append(
                Node(
                    i=i,
                    j=i + 1,
                    clip_id=None,
                    first_pos=None,
                    last_pos=None,
                    start_ms=None,
                    end_ms=None,
                    instance=FALLBACK,
                    target_cost=fallback_cost,
                    words=[target[i]],
                )
            )
    return Lattice(nodes=nodes, n=len(target))


@dataclass
class LatticeTile:
    words: list[str]
    length: int
    clip_id: int | None  # None => fallback (route through _pick_word_pick)
    start_ms: int | None
    end_ms: int | None
    variants: int = 0  # # clip instances that could fill this span (pre-prune)


def viterbi(lat: Lattice, join_cost, feats, rng) -> list[LatticeTile]:
    """Per-node min-cost path, minimizing (total cost, then unit count)
    lexicographically. f(node) = target_cost(node) + min over successors s
    (s.i == node.j) of [join_cost(node, s) + f(s)]; sink (node.j == n) has no
    forward join. The min is taken PER NODE because join cost couples a node to its
    successor -- this is what makes the search instance-level (spec section 5.2),
    not span-level. Source/sink boundary join cost = 0.

    Cost is PRIMARY; on EXACT cost ties the FEWER-unit (longer-run, fewer-join)
    path wins, so the lattice never gratuitously regresses joins vs greedy when the
    crude cost is indifferent. A strictly cheaper split still wins -- that is the
    lattice's job. Remaining exact ties are broken by seeded rng over a stable
    sort (reproducible). This makes H&B's 'longer runs emerge from zero-cost
    contiguity' robust even when the crude join cost is exactly zero."""
    assert lat.n > 0, "viterbi requires a non-empty target"
    by_start: dict[int, list[Node]] = {}
    for nd in lat.nodes:
        by_start.setdefault(nd.i, []).append(nd)

    INF = float("inf")
    # node -> (cost-to-end, units-to-end, next); units = #tiles from here to the end
    f: dict[Node, tuple[float, int, Node | None]] = {}
    # nodes starting at higher positions are resolved first
    for i in range(lat.n - 1, -1, -1):
        for nd in by_start.get(i, []):
            if nd.j == lat.n:  # sink: no forward join
                f[nd] = (nd.target_cost, 1, None)
                continue
            # (cost, units) TO END via each successor s (excludes nd's own target)
            cands = [
                (
                    join_cost.cost(nd.last_pos, s.first_pos, feats) + f[s][0],
                    1 + f[s][1],
                    s,
                )
                for s in by_start.get(nd.j, [])
            ]
            if not cands:  # dead end (fallback guarantees none)
                f[nd] = (INF, 1, None)
                continue
            cost_via, units, s = _pick_min(cands, rng)
            f[nd] = (nd.target_cost + cost_via, units, s)

    # source: minimize (cost, units) over start nodes (boundary join = 0)
    src = [(f[nd][0], f[nd][1], nd) for nd in by_start.get(0, [])]
    _, _, nd = _pick_min(src, rng)

    tiles: list[LatticeTile] = []
    while nd is not None:
        tiles.append(
            LatticeTile(
                words=nd.words,
                length=nd.j - nd.i,
                clip_id=nd.clip_id,
                start_ms=nd.start_ms,
                end_ms=nd.end_ms,
                variants=nd.n_candidates,
            )
        )
        nd = f[nd][2]
    # the path must tile the whole target; a gap would mean a broken DAG (a dead-end
    # source). Fail loudly rather than silently returning a truncated cover.
    assert sum(t.length for t in tiles) == lat.n, "viterbi produced an incomplete cover"
    return tiles


def _node_key(nd: Node):
    """Stable tie-break key (deterministic given the seed)."""
    return (
        nd.clip_id if nd.clip_id is not None else -1,
        nd.first_pos if nd.first_pos is not None else (-1, -1),
    )


def _pick_min(cands, rng):
    """From `cands` = [(cost, units, node), ...], return the entry minimizing
    (cost, then units) lexicographically — cost primary, fewer units (joins) wins a
    cost tie. Remaining exact ties are broken by seeded rng over a stable _node_key
    sort. The single point of truth for the Viterbi selection policy (used for both
    the per-node successor choice and the source choice)."""
    m_cost = min(c for c, _, _ in cands)
    cost_tied = [t for t in cands if abs(t[0] - m_cost) < 1e-12]
    m_units = min(u for _, u, _ in cost_tied)
    fully = sorted(
        (t for t in cost_tied if t[1] == m_units), key=lambda t: _node_key(t[2])
    )
    return rng.choice(fully)
