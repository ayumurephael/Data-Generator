"""One canonical crop, joint replay, explicit annotation and visibility."""
from dataclasses import asdict, dataclass
import math

import numpy as np

from . import catalog
from .planning import Episode, LOCAL
from .scm import world, solve


class NumericalFailure(Exception):
    pass


@dataclass
class Event:
    source: str
    event_id: int
    layer: str
    target: int
    kind: str
    start: int
    stop: int
    amplitude: object
    severity: object
    params: dict


def duration(r, kind, w, k, shortest, available=None):
    low = catalog.MIN_LENGTH[kind]
    high = min(w//4, max(low, w//(2*k)))
    if available is not None:
        high = min(high, available)
        if high < low:
            return 0
    if shortest or kind == "L07":
        return low
    return int(np.floor(.5 + np.exp(r.uniform(np.log(low), np.log(high)))))


def schedule(ep, bg, w, crop, lag, scale):
    if not ep.k:
        return [], "healthy"
    r = ep.rng("groups")
    k = ep.k
    nproc = k if ep.mode == "process" else 0 if ep.mode == "observation" else int(r.integers(1, k))
    layers = ["process"]*nproc + ["observation"]*(k-nproc)
    if ep.focus and layers[0] != ep.focus[0]:
        pos = layers.index(ep.focus[0])
        layers[0], layers[pos] = layers[pos], layers[0]
    possible = np.arange(ep.n)
    if ep.focus and ep.focus[1].startswith("S"):
        possible = [j for j in possible if ep.focus[1] in bg.eligible(int(j), crop+64, crop+w-64)]
        if not len(possible):
            raise RuntimeError(f"focal assignment {ep.focus} is incompatible with the frozen background of {ep.id}")
    first = int(r.choice(possible))
    sites = [first, *map(int, r.choice(np.delete(np.arange(ep.n), first), k-1, replace=False))]
    repeat = r.random(k) < .5
    pattern = str(r.choice(["simultaneous", "sequential", "overlap"])) if k > 1 else "single"
    # The fallback uses minimum lengths. Nonfocal seasonal choices depend on the
    # metadata at the final event interval; feasibility never consults outcomes.
    for attempt in range(51):
        r = ep.rng("geometry", attempt)
        shortest = attempt == 50
        groups = []
        for g, (site, layer) in enumerate(zip(sites, layers)):
            eventlist = []
            for pos in range(1+int(repeat[g])):
                focal = ep.focus if g == pos == 0 else None
                local = str(r.choice(LOCAL))
                if focal:
                    kind = focal[1]
                elif layer == "process" and r.random() < .5 and bg.eligible(site, crop, crop+w):
                    kind = "S*"  # uniformly select an eligible transform once the interval is known
                else:
                    kind = local
                d = duration(r, "S01" if kind == "S*" else kind, w, k, shortest)
                eventlist.append((kind, d, local))
            groups.append(eventlist)
        d = np.array([g[0][1] for g in groups])
        starts = np.zeros(k, dtype=int)
        if pattern == "sequential":
            cur = 0
            for g in r.permutation(k):
                starts[g] = cur
                cur += d[g] + int(r.integers(1+min(lag, w//(16*k))))
        elif pattern == "overlap":
            starts = -np.array([r.integers(length) for length in d])
            starts -= starts.min()
        end = int(max(starts+d))
        items = []
        for g, group in enumerate(groups):
            for pos, (kind, length, local) in enumerate(group):
                if pos:
                    length = duration(r, "S01" if kind == "S*" else kind, w, k, shortest, w-end)
                if length < catalog.MIN_LENGTH["S01" if kind == "S*" else kind]:
                    break
                start = int(starts[g]) if pos == 0 else end + int(r.integers(1 + min(lag, w-end-length)))
                items.append((g, pos, kind, length, start, local))
        if len(items) != k+int(repeat.sum()):
            continue
        latest = max(start+d for _, _, _, d, start, _ in items)
        if latest > w:
            continue
        last_shift = w-latest
        # Try shifts in a uniform permutation. Every feasible shift has equal
        # chance of being first even when a focal atom has very short support.
        for shift in r.permutation(last_shift+1):
            result = []
            for g, pos, kind, length, start, local in items:
                b = int(start+crop+shift)
                eligible = bg.eligible(sites[g], b, b+length) if kind[0] == "S" else ()
                if kind == "S*" and eligible:
                    kind = str(r.choice(sorted(eligible)))
                elif kind == "S*":
                    # No structure is compatible at this interval. Draw the local
                    # duration from its own law (including single-point outliers).
                    kind = local
                    new_duration = duration(r, kind, w, k, shortest)
                    if new_duration > length:
                        break
                    length = new_duration
                elif kind[0] == "S" and kind not in eligible:
                    break
                severity = amplitude = None
                if kind.startswith("L") or kind == "S04":
                    s = ep.focus[2] if ep.focus and g == pos == 0 and ep.focus[2] >= 0 else int(r.integers(3))
                    severity = ("weak", "medium", "strong")[s]
                    a, z = ((.1, .5), (.5, 2.), (2., 6.))[s]
                    amplitude = float(scale*r.uniform(a, z))
                result.append(Event(str(g), pos, layers[g], sites[g], kind, b, b+length, amplitude,
                                    severity, {"stream": f"event/{g}/{pos}/operator"}))
            if len(result) == len(items):
                return result, pattern
    raise RuntimeError(f"no geometrically valid schedule for {ep.id} focus={ep.focus}")


def hysteresis(magnitude):
    labels = np.zeros(len(magnitude), dtype=bool)
    active = False
    quiet = 0
    for t, value in enumerate(magnitude):
        if not active and value >= .5:
            active = True
        if active:
            labels[t] = True
            quiet = quiet + 1 if value <= .25 else 0
            if quiet == 3:
                labels[t-2:t+1] = False
                active = False
                quiet = 0
    return labels, active


def observe(ep, w, ordinary, crop, normal, reading, delta, direct, scale, y, events=()):
    n = ep.n
    r = ep.rng("acquisition")
    mode = str(r.choice(["complete", "independent", "gaps"], p=[.35, .25, .4]))
    observed = np.ones((ordinary,n), dtype=bool)
    if mode == "independent":
        observed = r.random((ordinary,n)) >= r.uniform(0, .15, (1,n))
    if mode == "gaps":
        for i in range(n):
            for _ in range(int(r.integers(1,4))):
                d = int(np.floor(.5+np.exp(r.uniform(np.log(2),np.log(max(2,w//4))))))
                start = int(r.integers(ordinary-d+1))
                observed[start:start+d,i] = False
    r = ep.rng("thinning")
    idx = np.arange(crop, crop+w)
    if r.random() < .25:
        keep = r.random(w) < r.uniform(.65,.95)
        keep[[0,-1]] = True
        idx = idx[keep]
    med = np.zeros(n)
    scales = np.ones(n)
    for i in range(n):
        x = reading[:64,i][observed[:64,i]]
        if len(x):
            q1, med[i], q3 = np.quantile(x, [.25,.5,.75])
            if q3-q1 >= .001:
                scales[i] = max(.001,(q3-q1)/1.349)
    mask = observed[idx]
    def normalize(values):
        out = np.clip((values[idx]-med)/scales, -30., 30.).astype(np.float32)
        out[~mask] = 0.
        return out
    values = normalize(reading)
    visible = mask & (abs(values.astype(np.float64)-normalize(normal)) > 1e-5)
    witness = visible & (direct[idx] | (delta[idx] >= .25*scale))
    root = direct[crop:crop+w].any(axis=0)
    known_root = (visible & direct[idx]).any(axis=0)
    rk = (~root | known_root) if (not root.any() or known_root.any()) else np.zeros(n, dtype=bool)
    valid = np.ones(n,dtype=bool)
    inputs = dict(values=values, observed=mask, time=idx.astype(float), channel_valid=valid)
    targets = dict(Y=y[idx], Y_known=~y[idx] | witness.any(axis=1), R=root, R_known=rk, E=valid)
    event_witness = [bool(visible[(idx >= e.start) & (idx < e.stop), e.target].any()) for e in events]
    return inputs, targets, {"acquisition": mode, "prefix_center": med.tolist(), "prefix_scale": scales.tolist(),
                             "event_witness": event_witness,
                             "no_visible_root": bool(root.any() and not known_root.any()),
                             "only_propagation_visible": bool(witness.any() and not (visible & direct[idx]).any()),
                             "no_time_witness": bool(y[crop:crop+w].any() and not (y[idx] & witness.any(1)).any()),
                             "all_positive_rows_thinned": bool(y[crop:crop+w].any() and not y[idx].any())}


def healthy_path(wld, bg):
    drive = np.ascontiguousarray(bg.b, dtype=np.float64)
    return solve(drive, np.empty((0,0)), False, *wld.args)[0]


def realize(ep, scale, retry=0, audit=False):
    wld = world(ep.split, ep.n, ep.family, ep.ancestor)
    w, ordinary, crop = ep.clock(wld.lag)
    frozen = catalog.normal_seasons(ep.rng("background"),ep.n,wld.warm,ordinary) if retry else None
    def background(extra=0):
        return catalog.normal_background(ep.rng("background",retry),ep.n,wld.warm,ordinary,
                                         extra=extra,frozen_seasons=frozen)
    bg = background()
    events, pattern = schedule(ep, bg, w, crop, wld.lag, scale)
    extra = 0
    censored = False
    while True:
        shape = bg.b.shape
        proc = np.zeros(shape)
        q = np.zeros(shape)
        direct = np.zeros((shape[0]-wld.warm, ep.n), dtype=bool)
        for event in events:
            b, e, i = event.start, event.stop, event.target
            span = slice(wld.warm+b, wld.warm+e)
            r = ep.rng(event.params["stream"])
            direct[b:e,i] = True
            if event.kind[0] == "S":
                proc[span,i] += bg.season_delta(i, event.kind, b, e, r, event.amplitude)
            else:
                signal = catalog.local_wave(event.kind,e-b,r,event.amplitude)
                (q if event.layer == "observation" else proc)[span,i] += signal
        if (proc != 0.).any():
            x0, xj, upper = solve(np.ascontiguousarray(bg.b), proc, True, *wld.args)
        else:
            x0 = healthy_path(wld, bg)
            xj, upper = x0, np.zeros_like(x0)
        x0, xj, upper, q = (x[wld.warm:] for x in (x0,xj,upper,q))
        reading = xj + q
        delta = abs(xj-x0)
        if not (np.isfinite(x0).all() and np.isfinite(reading).all() and np.isfinite(upper).all()):
            raise NumericalFailure("nonfinite")
        if max(np.max(abs(x0[:ordinary])), np.max(abs(xj[:ordinary])),np.max(abs(reading[:ordinary]))) > 256.:
            raise NumericalFailure("range")
        if np.max(x0[:ordinary].std(axis=0)) < .001:
            raise NumericalFailure("flat")
        if np.any(delta-upper > 1e-10*(1+upper)):
            raise ArithmeticError("structural perturbation exceeds proven lag budget")
        last = max((e.stop for e in events), default=0)
        margin = .25*scale
        # Include three physical quiet ticks after the contraction certificate.
        t = len(delta)-3
        history = delta[max(0,t-wld.lag):t]
        tail = float(history.max(initial=0.))
        if t >= last and tail <= margin:
            break
        if extra == 65536:
            censored = True
            break
        rho = wld.budget.sum(axis=1).max()
        bound = 0 if tail <= margin else (wld.lag if rho == 0. else
            wld.lag*math.ceil(math.log(tail/margin)/-math.log(rho)))
        extra = min(65536, extra+max(3, last+3-t, bound+3))
        bg = background(extra)
    h, active = hysteresis(delta.max(axis=1)/scale)
    y = h | direct.any(axis=1)
    inputs, targets, private = observe(ep,w,ordinary,crop,x0,reading,delta,direct,scale,y,events)
    after = max((e.stop for e in events),default=len(h))
    onset = np.flatnonzero(h & ~np.r_[False,h[:-1]])
    private["new_propagation_onset_after_last_event"] = bool((onset >= after).any())
    if retry:
        private["season_retry"] = 0
    assert int(targets["R"].sum()) == ep.k and np.all(np.diff(inputs["time"]) > 0)
    private.update(id=ep.id, N=ep.n, family=ep.family, ancestor=ep.ancestor, episode=ep.episode,
                   K=ep.k, mode=ep.mode, split=ep.split, focal=list(ep.focus) if ep.focus else None,
                   T=len(inputs["time"]), L=wld.lag, w=w, crop_start=crop, T_grid=len(delta),
                   scale=scale, extra_guard=extra, right_censored=bool(censored or active),
                   events=[asdict(e) for e in events], layout=pattern, retry=retry,
                   max_bound_excess=float((delta-upper).max()), max_row_excess=float((wld.budget.sum(1)-wld.rho).max()))
    traces = (dict(normal=x0, factual=xj, reading=reading, delta=delta, upper=upper,
                   direct=direct, y=y) if audit else None)
    return inputs, targets, private, traces


def require_frozen_seasons(record,retries):
    """Pre-fix first attempts are equivalent. Retried archives require explicit provenance."""
    if retries and record.get("season_retry") != 0:
        raise ValueError(f"legacy numerical retry in {record['id']}: rebuild its shard and revalidate width exposure")
