"""Sparse lagged structural equations. Paired paths use exactly the same drives."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math

import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(**kwargs):
        return lambda fn: fn

PROFILE = "sster-direct-state-20261125-v1"
FAMILIES = ("linear", "additive", "interaction", "saturating")
WIDTHS = (1, 4, 8, 16, 32, 64, 128)


def rng_for(*parts):
    digest = hashlib.sha256("|".join(map(str, (PROFILE, *parts))).encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "big")))


def stream(split, n, family, ancestor, episode, component, retry=0):
    return rng_for(split, n, family, ancestor, episode, component, retry)


def bucket(n, family, candidate):
    text = f"{PROFILE}|generic|{n}|{family}|{candidate}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest(), "big") % 100


BUCKETS = {"train": (0, 80), "validation": (80, 90), "test": (90, 100)}


@lru_cache(maxsize=128)
def reserved_ancestors(split, n, family, count):
    low, high = BUCKETS[split]
    ids = []
    i = 0
    while len(ids) < count:
        if low <= bucket(n, family, i) < high:
            ids.append(i)
        i += 1
    return tuple(ids)


def ancestors(split, n, family, count):
    """Effective IDs in reserved-slot order, including any deterministic replacements."""
    from .ancestry import resolve
    return list(resolve(split,n,(family,),count)[family].ids)


class InvalidAncestor(Exception):
    """The deterministic SCM draw violates a structural invariant before any episode."""


@dataclass
class World:
    lag: int
    warm: int
    budget: np.ndarray
    rho: np.ndarray
    args: tuple


def signed(rng, size=None):
    return rng.choice(np.array([-1., 1.]), size=size) * rng.uniform(.1, 1., size)


@lru_cache(maxsize=8)
def world(split, n, family, ancestor):
    r = stream(split, n, family, ancestor, 0, "world/graph")
    lag = int(r.integers(1, 17))
    parents = np.zeros((n, 5), dtype=np.int64)
    delays = parents.copy()
    count = np.zeros(n, dtype=np.int64)
    typ = parents.copy()
    coeff = np.zeros((n, 5), dtype=float)
    params = np.zeros((n, 5, 7), dtype=float)
    pair = np.zeros((n, 2), dtype=np.int64)
    interaction = np.zeros(n)
    units = np.zeros(n, dtype=np.int64)
    weights = np.zeros((n, 8, 5))
    bias, beta = np.zeros((n, 8)), np.zeros((n, 8))
    saturation = np.zeros(n, dtype=np.int64)
    cut = np.zeros((n, 2))
    budgets = np.zeros_like(coeff)
    rho = np.zeros(n)
    for i in range(n):
        r = stream(split, n, family, ancestor, 0, f"world/node/{i}")
        available = np.delete(np.arange(n), i)
        others = r.choice(available, size=int(r.integers(min(n - 1, 4) + 1)), replace=False)
        edges = [(j, int(r.integers(1, lag + 1))) for j in others]
        if r.random() < .8:
            edges.append((i, 1))
        edges.sort()
        p = len(edges)
        count[i] = p
        limit = rho[i] = r.uniform(.1, .85)
        reserve = r.uniform(.6, 1.)
        if family == "saturating":
            saturation[i] = r.integers(1, 3)
            cut[i] = r.uniform(1., 4., 2)
        if not p:
            continue
        parents[i, :p], delays[i, :p] = zip(*edges)
        inside = family
        nonlinear = ["mlp", "product"] if p >= 2 else ["mlp"]
        if family == "saturating":
            inside = r.choice(["linear", "additive", *nonlinear])
        if family == "interaction":
            inside = r.choice(nonlinear)
        for k in range(p):
            coeff[i, k] = signed(r)
            code = int(r.integers(7)) if inside != "linear" else 0
            typ[i, k] = code
            if code == 6:
                slopes = [(0, 1, 1), (1, 0, 1), (0, 1, 0), r.uniform(0, 1, 3)][int(r.integers(4))]
                params[i, k, :5] = [r.uniform(-1.5, -.1), r.uniform(.1, 1.5), *slopes]
            elif code:
                theta, gamma = r.uniform(-1, 1), np.exp(r.uniform(np.log(.5), np.log(2.)))
                params[i, k, :3] = (theta, gamma, prototype(code, -gamma * theta))
        if inside == "product":
            pair[i] = r.choice(p, 2, replace=False)
            interaction[i] = signed(r)
        if inside == "mlp":
            h = units[i] = r.choice([2, 4, 8])
            weights[i, :h, :p] = r.normal(0, 1 / np.sqrt(p), (h, p))
            bias[i, :h] = r.uniform(-1., 1., h)
            beta[i, :h] = signed(r, h) / np.sqrt(h)
        b = np.abs(coeff[i]) + np.abs(beta[i, :, None] * weights[i]).sum(axis=0)
        if inside == "product":
            b[pair[i]] += abs(interaction[i])
        factor = reserve * limit / b.sum()
        coeff[i] *= factor
        interaction[i] *= factor
        beta[i] *= factor
        budgets[i] = b * factor
    if not np.all(np.isfinite(rho) & (rho >= .1) & (rho <= .85)):
        raise InvalidAncestor("row-contraction-rate")
    row_budget = budgets.sum(axis=1)
    if not np.all(np.isfinite(row_budget) & (row_budget <= rho + 1e-12)):
        raise InvalidAncestor("row-contraction-budget")
    warm = max(64, lag * math.ceil(math.log(1e-6) / math.log(max(rho))))
    args = (count, parents, delays, typ, coeff, params, pair, interaction, units,
            weights, bias, beta, np.tanh(bias), saturation, cut, budgets)
    return World(lag, warm, budgets, rho, args)


@njit(cache=True)
def prototype(code, x):
    if code == 1:
        return math.sin(x)
    if code == 2:
        return math.tanh(x)
    if code == 3:
        return x / (1. + abs(x))
    if code == 4:
        return .5 * math.tanh(x) ** 2
    if code == 5:
        return math.tanh(x) ** 3 / 3.
    return x


@njit(cache=True)
def response(code, x, par):
    if code == 0:
        return x
    if code == 6:
        a, b, k0, k1, k2 = par[:5]
        return k0*x + (k1-k0)*(max(x-a, 0.)-max(-a, 0.)) + (k2-k1)*(max(x-b, 0.)-max(-b, 0.))
    return (prototype(code, par[1]*(x-par[0]))-par[2])/par[1]


@njit(cache=True)
def saturate(x, kind, cut):
    if kind == 1:
        return max(-cut[0], min(cut[1], x))
    if kind == 2:
        q = cut[int(x >= 0.)]
        return q * math.tanh(x / q)
    return x


@njit(cache=True)
def solve(drive, injection, paired, count, parent, delay, typ, coeff, par, pair, cross,
          units, weights, bias, beta, tanh_bias, sat, cut, budget):
    steps, n = drive.shape
    normal = np.zeros_like(drive)
    factual = np.zeros_like(drive) if paired else np.empty((0, 0))
    upper = np.zeros_like(drive) if paired else np.empty((0, 0))
    zs = np.zeros(5)
    for t in range(steps):
        for i in range(n):
            direct_normal = 0.
            limit = 0.
            for branch in range(1 + int(paired)):
                base = drive[t, i]
                for k in range(count[i]):
                    last = t - delay[i, k]
                    j = parent[i, k]
                    z = 0.
                    if last >= 0:
                        z = normal[last, j] if branch == 0 else factual[last, j]
                        if branch:
                            limit += budget[i, k] * upper[last, j]
                    zs[k] = z
                    base += coeff[i, k] * response(typ[i, k], z, par[i, k])
                if cross[i] != 0.:
                    base += cross[i] * math.tanh(zs[pair[i, 0]]) * math.tanh(zs[pair[i, 1]])
                for h in range(units[i]):
                    total = bias[i, h]
                    for k in range(count[i]):
                        total += weights[i, h, k] * zs[k]
                    base += beta[i, h] * (math.tanh(total) - tanh_bias[i, h])
                if branch == 0:
                    normal[t, i] = saturate(base, sat[i], cut[i])
                    if paired:
                        direct_normal = abs(saturate(base + injection[t, i], sat[i], cut[i]) - normal[t, i])
                else:
                    factual[t, i] = saturate(base + injection[t, i], sat[i], cut[i])
                    upper[t, i] = limit + direct_normal
    return normal, factual, upper
