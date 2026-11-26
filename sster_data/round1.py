"""Select the first-round corpus from the full deterministic ticket schedule."""
from pathlib import Path
import os
import shutil

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow

from .ancestry import check_plan, ledger, sidecar
from .planning import LOCAL, MODES, SEASONAL, plan, templates
from .scm import FAMILIES, WIDTHS, rng_for
from .storage import digest, lock, read_json, write_json

EDITION = "round1-1680k-v1"
ANCESTORS = 3750
PER_ANCESTOR = 16
FOCAL = 48
WIDTH_TOTAL = 240000
_BASE = (4500, 27950, 28200, 28950, 39200, 39200)
_COUNTS = {1: (168000,), 4: (5500, 56250, 55000, 51250),
           64: (4496, 27946, 28200, 28950, 39204, 39204),
           128: (4504, 27954, 28200, 28950, 39196, 39196)}


def is_first_round(root):
    tag = root / "build" / "training-edition.json"
    if not tag.exists():
        return False
    if read_json(tag) != {"training_edition": EDITION}:
        raise ValueError("unknown training edition on disk")
    return True


def bind(root, edition):
    if is_first_round(root):
        if edition != EDITION:
            raise ValueError("first-round data in output directory; specify its training edition")
    elif edition == EDITION:
        for path in (root/"build").glob("plan-*.npy"):
            width = int(path.stem.split("-")[-1])
            if not (path.parent/f"round1-{width}.json").is_file():
                raise ValueError("cannot adopt an existing full-size plan as a first-round release")
        if (root/"_SUCCESS").exists():
            raise ValueError("cannot adopt a published unbound split")
        write_json(root/"build"/"training-edition.json", {"training_edition": EDITION})
    elif any((root/"build").glob("round1-*.json")):
        raise ValueError("first-round plans require the matching training edition")


def quotas():
    counts = {(n, k): m for n in WIDTHS
              for k, m in enumerate(_COUNTS.get(n, _BASE), 1)}
    mixed = {key: (m * 3 // 25 if key[1] >= 2 else 0) for key, m in counts.items()}
    eligible = sorted((key for key in counts if key[1] >= 2),
                      key=lambda key: (-(counts[key]*3 % 25), key))
    extras = 117600 - sum(mixed.values())
    if not 0 <= extras <= len(eligible):
        raise ValueError("invalid mixed quota")
    for key in eligible[:extras]:
        mixed[key] += 1
    result = {(n, 0, "healthy"): 72000 for n in WIDTHS}
    for (n, k), total in counts.items():
        m = mixed[n, k]
        for mode, size in zip(MODES[1:], (total//2, total-total//2-m, m)):
            if size:
                result[n, k, mode] = size
    return result


def _key(k, mode, layer, kind, severity):
    return ((((k * 4 + mode) * 3 + layer + 1) * 41 + kind + 1) * 4 + severity + 1)


def keys(records):
    return _key(*(records[field].astype(np.int32) for field in
                  ("k", "mode", "layer", "kind", "severity")))


def target_counts(n):
    result = {}
    for (width, k, mode), count in quotas().items():
        if width != n:
            continue
        mode_id = MODES.index(mode)
        for layer, kind in (templates(mode) if k else ()):
            layer_id = ("process", "observation").index(layer)
            template = (LOCAL + SEASONAL).index(kind)
            bands = (0, 1, 2) if kind in LOCAL or kind == "S04" else (-1,)
            for severity in bands:
                result[_key(k, mode_id, layer_id, template, severity)] = FOCAL // len(bands)
            count -= FOCAL
        if count < 0:
            raise ValueError("specified focal examples exceed cell size")
        result[_key(k, mode_id, -1, -1, -1)] = count
    return result


def select_tickets(codes, demand, candidates_per_ancestor, take, rng):
    """Integral flow enforces exact per-ancestor and per-stratum counts simultaneously."""
    codes = np.asarray(codes, dtype=np.int32)
    demand = np.asarray(demand, dtype=np.int32)
    available = np.bincount(codes, minlength=len(demand))
    ancestors, remainder = divmod(len(codes), candidates_per_ancestor)
    if (remainder or not 0 < take <= candidates_per_ancestor or demand.sum() != ancestors*take
            or np.any(demand < 0) or np.any(demand > available)):
        raise ValueError("infeasible ticket totals")
    # Start near the intended distribution within each structural world. The flow
    # changes only enough stratum memberships to meet exact population totals.
    priority = rng.random(len(codes))
    rate = demand[codes] / available[codes]
    rank = np.divide(priority, rate, out=np.full_like(priority, np.inf), where=rate > 0)
    initial = np.zeros((ancestors, candidates_per_ancestor), dtype=bool)
    first = np.argpartition(rank.reshape(initial.shape), take-1, axis=1)[:, :take]
    np.put_along_axis(initial, first, True, axis=1)
    nstrata = len(demand)
    shuffled = np.repeat(rng.permutation(ancestors), candidates_per_ancestor)
    pairs, inverse, capacity = np.unique(shuffled*nstrata + codes, return_inverse=True, return_counts=True)
    used = np.bincount(inverse[initial.ravel()], minlength=len(pairs))
    a, c = np.divmod(pairs, nstrata)
    observed = np.bincount(codes[initial.ravel()], minlength=nstrata)
    surplus, deficit = np.maximum(observed-demand, 0), np.maximum(demand-observed, 0)
    source, sink = ancestors+nstrata, ancestors+nstrata+1
    strata = ancestors + np.arange(nstrata)
    rows = np.concatenate((a, ancestors+c, np.full(nstrata, source), strata))
    cols = np.concatenate((ancestors+c, a, strata, np.full(nstrata, sink)))
    caps = np.concatenate((capacity-used, used, surplus, deficit))
    positive = caps > 0
    graph = coo_matrix((caps[positive], (rows[positive], cols[positive])),
                       shape=(sink+1, sink+1)).tocsr()
    flow = maximum_flow(graph, source, sink)
    if flow.flow_value != deficit.sum():
        raise ValueError("no feasible joint assignment of ancestors and template quotas")
    wanted = used + np.asarray(flow.flow[a, ancestors+c]).reshape(-1)
    ordered = np.lexsort((priority, inverse))
    start = np.cumsum(capacity) - capacity
    position = np.arange(len(codes)) - np.repeat(start, capacity)
    chosen = np.sort(ordered[position < np.repeat(wanted, capacity)]).astype(np.int32)
    if (not np.array_equal(np.bincount(codes[chosen], minlength=nstrata), demand)
            or not np.all(np.bincount(chosen//candidates_per_ancestor, minlength=ancestors) == take)):
        raise ArithmeticError("selected rows violate exact joint quotas")
    return chosen


def _save(path, rows):
    pending = path.with_suffix(f".{os.getpid()}.tmp")
    with pending.open("wb") as f:
        np.save(f, rows, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(pending, path)


def _freeze(path, original):
    if not path.exists():
        temp = path.with_suffix(f".{os.getpid()}.tmp")
        shutil.copyfile(original, temp)
        os.replace(temp, path)
    if digest(path) != digest(original):
        raise ValueError(f"frozen parent schedule changed: {original}")


def reuse_calibration(out, parent):
    original = Path(parent) / "calibration.json"
    if digest(original) != original.with_suffix(".sha256").read_text(encoding="ascii").strip():
        raise IOError("invalid parent calibration checksum")
    dest = Path(out) / "calibration.json"
    with lock(dest.with_suffix(".lock")):
        _freeze(dest, original)
        _freeze(dest.with_suffix(".sha256"), original.with_suffix(".sha256"))


def prepare(path, n, roster, parent=None):
    base = path.parent / "parent"
    base.mkdir(exist_ok=True, parents=True)
    source = base / path.name
    if parent is not None:
        old = Path(parent) / "train" / "build" / path.name
        _freeze(source, old)
        if sidecar(old).exists():
            _freeze(sidecar(source), sidecar(old))
    parent_plan = plan(source, "train", n, False, roster)
    evidence = path.parent / f"round1-{n}.json"
    if evidence.exists():
        return verify(path, n, roster)
    wanted = target_counts(n)
    strata, codes = np.unique(keys(parent_plan), return_inverse=True)
    if set(wanted) != set(strata):
        raise ValueError("full schedule does not offer exactly the requested strata")
    target = [wanted[key] for key in strata]
    rows = select_tickets(codes, target, 64, PER_ANCESTOR, rng_for(EDITION, n, "select"))
    selected_path = path.parent / f"source-rows-{n}.npy"
    _save(selected_path, rows)
    _save(path, parent_plan[rows])
    write_json(sidecar(path), ledger("train", n, False, roster, digest(path)))
    write_json(evidence, dict(training_edition=EDITION, parent_sha256=digest(source),
                              plan_sha256=digest(path), source_rows_sha256=digest(selected_path)))
    return verify(path, n, roster)


def verify(path, n, roster):
    source = path.parent / "parent" / path.name
    parent = check_plan(source, "train", n, False, roster)
    selected = path.parent / f"source-rows-{n}.npy"
    sha = digest(path)
    if read_json(path.parent/f"round1-{n}.json") != dict(training_edition=EDITION,
            parent_sha256=digest(source), plan_sha256=sha, source_rows_sha256=digest(selected)):
        raise ValueError("first-round plan binding mismatch")
    if read_json(sidecar(path)) != ledger("train", n, False, roster, sha):
        raise ValueError("first-round ancestor binding mismatch")
    rows = np.load(selected, mmap_mode="r", allow_pickle=False)
    p = np.load(path, mmap_mode="r", allow_pickle=False)
    if (rows.dtype != np.int32 or rows.shape != (WIDTH_TOTAL,) or np.any(np.diff(rows) <= 0)
            or rows[0] < 0 or rows[-1] >= len(parent)
            or not np.all(np.bincount(rows//64, minlength=ANCESTORS*len(FAMILIES)) == PER_ANCESTOR)
            or p.dtype != parent.dtype or not np.array_equal(p, parent[rows])):
        raise ValueError("first-round plan is not an exact subset of the parent tickets")
    k, counts = np.unique(keys(p), return_counts=True)
    if dict(zip(k, counts)) != target_counts(n):
        raise ValueError("first-round severity, template or cell quotas differ")
    return p
