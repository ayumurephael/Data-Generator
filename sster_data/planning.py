"""Quota allocation and pre-injection matching on independently drawn normal backgrounds."""
from dataclasses import dataclass
import hashlib
from itertools import chain
from pathlib import Path
import os

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow

from .ancestry import check_plan, ledger, resolve, sidecar
from .scm import FAMILIES, WIDTHS, PROFILE, rng_for, stream
from .storage import digest, read_json, write_json

MODES = ("healthy", "process", "observation", "mixed")
LOCAL = tuple(f"L{i:02}" for i in range(1, 21))
SEASONAL = tuple(f"S{i:02}" for i in range(1, 21))
LAYERS = ("process", "observation")
PRODUCTION = dict(train=(3750, 64, 96), validation=(750, 16, 4), test=(750, 16, 4))
PLAN_TYPE = np.dtype([("family", "u1"), ("ancestor", "<i4"), ("episode", "u1"),
                      ("k", "u1"), ("mode", "u1"), ("layer", "i1"), ("kind", "i1"), ("severity", "i1")])


def dimensions(split, smoke):
    return (1, 4, 0) if smoke else PRODUCTION[split]


def widths(smoke):
    return (1, 4, 8) if smoke else WIDTHS


def quotas(split, smoke=False):
    if smoke:
        return {(n, k, mode): count for n in widths(True)
                for k, mode, count in ([(0, "healthy", 4), (1, "process", 6), (1, "observation", 6)]
                if n == 1 else [(0, "healthy", 4), (1, "process", 3), (2, "observation", 3),
                                (min(6, n), "mixed", 6)])}
    sizes = {1: [672000], 4: [22000, 225000, 220000, 205000]}
    divisor = 1 if split == "train" else 20
    counts = {(n, k): value // divisor for n in WIDTHS for k, value in enumerate(
        sizes.get(n, [18000, 111800, 112800, 115800, 156800, 156800]), 1)}
    mixed = {key: m*3//25 if key[1] >= 2 else 0 for key, m in counts.items()}
    order = sorted((key for key in counts if key[1] >= 2), key=lambda key: (-(counts[key]*3 % 25), key))
    extras = sum(counts[key]*3 for key in order)//25 - sum(mixed.values())
    for key in order[:extras]:
        mixed[key] += 1
    out = {(n, 0, "healthy"): 288000 // divisor for n in WIDTHS}
    for key, m in counts.items():
        for mode, value in zip(MODES[1:], (m//2, m-m//2-mixed[key], mixed[key])):
            if value:
                out[(*key, mode)] = value
    return out


def encode(family, ancestor, episode, k, mode, layer, kind, severity):
    return (FAMILIES[int(family)], int(ancestor), int(episode), int(k), MODES[int(mode)],
            None if layer < 0 else (LAYERS[int(layer)], (LOCAL + SEASONAL)[int(kind)], int(severity)))


@dataclass(frozen=True)
class Episode:
    split: str
    n: int
    family: str
    ancestor: int
    episode: int
    k: int
    mode: str
    focus: object
    smoke: bool = False

    def rng(self, name, retry=0):
        return stream(self.split, self.n, self.family, self.ancestor, self.episode, "episode/"+name, retry)

    @property
    def id(self):
        key = (PROFILE, "smoke" if self.smoke else "full", self.split, self.n, self.family, self.ancestor, self.episode)
        return hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()[:32]

    def clock(self, lag):
        r = self.rng("crop")
        w = int(r.choice((256,) if self.smoke else (256, 512, 1024, 2048)))
        length = 64 + 2*w + max(3*lag, 3)
        return w, length, int(r.integers(64, length-w+1))


def templates(mode):
    if mode in ("process", "mixed"):
        yield from (("process", s) for s in LOCAL + SEASONAL)
    if mode in ("observation", "mixed"):
        yield from (("observation", s) for s in LOCAL)


def match_seasons(eligible, reps):
    """Integer max flow. A capacity-one candidate cannot cover two tickets."""
    m, kinds = eligible.shape
    src, sink = m+kinds, m+kinds+1
    rows, cols = np.nonzero(eligible.T)
    r = np.concatenate((np.full(kinds, src), rows, kinds + np.arange(m)))
    c = np.concatenate((np.arange(kinds), kinds + cols, np.full(m, sink)))
    v = np.concatenate((np.full(kinds, reps), np.ones(len(rows)+m, dtype=np.int64)))
    graph = coo_matrix((v, (r, c)), shape=(sink+1, sink+1)).tocsr()
    result = maximum_flow(graph, src, sink)
    if result.flow_value != reps * kinds:
        return None
    matches = result.flow[:kinds, kinds:kinds+m].tocoo()
    return [(int(j), int(i)) for i, j, v in zip(matches.row, matches.col, matches.data) if v > 0]


def cover(records, ids, split, n, mode, reps):
    from .catalog import normal_seasons
    pool = list(templates(mode))
    used = set()
    offset = 0
    if mode != "observation":
        masks = []
        while offset < len(ids):
            end = min(len(ids), max(offset*2, int(1.5*len(pool)*reps)))
            for idx in ids[offset:end]:
                ep = Episode(split, n, *encode(*records[idx]))
                # Only the graph lag enters the crop clock. Season parameters
                # depend on ordinary duration; no trajectory needs rendering.
                lag = int(stream(split, n, ep.family, ep.ancestor, 0, "world/graph").integers(1,17))
                w, length, crop = ep.clock(lag)
                season = normal_seasons(ep.rng("background"), n, 0, length)
                eligible = set()
                for channel in range(n):
                    eligible.update(season.eligible(channel, crop+64, crop+w-64))
                    if len(eligible) == len(SEASONAL):
                        break
                masks.append([kind in eligible for kind in SEASONAL])
            offset = end
            matrix = np.array(masks, dtype=bool)
            order = np.argsort(matrix.sum(axis=0), kind="stable")
            matches = match_seasons(matrix[:, order], reps)
            if matches is not None:
                counts = [0] * len(SEASONAL)
                for j, code in matches:
                    code = int(order[code])
                    idx = int(ids[j])
                    used.add(idx)
                    records["layer"][idx] = 0
                    records["kind"][idx] = 20 + code
                    records["severity"][idx] = counts[code] % 3 if code == 3 else -1
                    counts[code] += 1
                break
        else:
            raise RuntimeError(f"no full seasonal coverage matching in {split}, N={n}, {mode}")
    free = (int(j) for j in ids if int(j) not in used)
    for layer, kind in pool:
        if kind in SEASONAL:
            continue
        for num in range(reps):
            idx = next(free)
            records["layer"][idx] = LAYERS.index(layer)
            records["kind"][idx] = LOCAL.index(kind)
            records["severity"][idx] = num % 3


def plan(path, split, n, smoke, rosters=None):
    a, episodes, reps = dimensions(split, smoke)
    if rosters is None:
        rosters = resolve(split,n,FAMILIES,a)
    if path.exists():
        return check_plan(path,split,n,smoke,rosters)
    records = np.zeros(4*a*episodes, dtype=PLAN_TYPE)
    records["layer"] = records["kind"] = records["severity"] = -1
    i = 0
    for family_id, family in enumerate(FAMILIES):
        for candidate in rosters[family].ids:
            ix = slice(i, i+episodes)
            records["family"][ix] = family_id
            records["ancestor"][ix] = candidate
            records["episode"][ix] = np.arange(episodes)
            i += episodes
    order = rng_for(split, n, "quota").permutation(len(records))
    offset = 0
    for (width, k, mode), count in quotas(split, smoke).items():
        if width != n:
            continue
        ids = order[offset:offset+count]
        records["k"][ids], records["mode"][ids] = k, MODES.index(mode)
        if reps and k:
            cover(records, ids, split, n, mode, reps)
        offset += count
    assert offset == len(records)
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    with open(temp, "wb") as out:
        np.save(out, records)
        out.flush()
        os.fsync(out.fileno())
    expected = ledger(split,n,smoke,rosters,digest(temp))
    binding = sidecar(path)
    if binding.exists() and read_json(binding) != expected:
        raise ValueError(f"orphaned ancestor replacement ledger disagrees with regenerated plan: {binding}")
    write_json(binding,expected)
    os.replace(temp,path)
    return check_plan(path,split,n,smoke,rosters)
