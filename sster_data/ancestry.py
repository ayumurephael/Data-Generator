"""C.4 deterministic admission of structural worlds, before assigning any episode ticket."""
from dataclasses import dataclass
import math

import numpy as np

from . import scm
from .storage import digest, read_json


def validate_world(split, n, family, candidate):
    w = scm.world(split, n, family, candidate)
    count, parent, lag, typ, coef, par, pair, cross, units, weights, bias, beta, tbias, sat, cut, budget = w.args
    # A changed implementation ABI is an error independent of the random draw.
    shapes = ((count,(n,)), (parent,(n,5)), (lag,(n,5)), (typ,(n,5)), (coef,(n,5)),
              (par,(n,5,7)), (pair,(n,2)), (cross,(n,)), (units,(n,)), (weights,(n,8,5)),
              (bias,(n,8)), (beta,(n,8)), (tbias,(n,8)), (sat,(n,)), (cut,(n,2)),
              (budget,(n,5)), (w.budget,(n,5)), (w.rho,(n,)))
    if any(a.shape != shape for a,shape in shapes):
        raise ValueError("incompatible SCM parameter array shape")
    for a in (count,parent,lag,typ,pair,units,sat):
        if not np.issubdtype(a.dtype,np.integer):
            raise TypeError("SCM structural indices must be integers")
    if any(not np.isfinite(a).all() for a in (coef,par,cross,weights,bias,beta,tbias,cut,budget,w.budget,w.rho)):
        raise scm.InvalidAncestor("nonfinite-structural-parameter")
    if not 1 <= w.lag <= 16 or np.any((w.rho < .1) | (w.rho > .85)):
        raise scm.InvalidAncestor("lag-or-contraction-rate")
    if w.warm != max(64, w.lag*math.ceil(math.log(1e-6)/math.log(float(w.rho.max())))):
        raise scm.InvalidAncestor("warmup-bound")
    active = np.arange(5)[None,:] < count[:,None]
    if np.any((count < 0) | (count > min(n,5))) or np.any(active & (
            (parent < 0) | (parent >= n) | (lag < 1) | (lag > w.lag))):
        raise scm.InvalidAncestor("parent-or-lag")
    self_edge = active & (parent == np.arange(n)[:,None])
    if (np.any(self_edge & (lag != 1)) or np.any((active & ~self_edge).sum(1) > 4)
            or np.any(active[:,1:] & (np.diff(parent,axis=1) <= 0))):
        raise scm.InvalidAncestor("duplicate-parent-or-self-lag")
    if (np.any(active & ((typ < 0) | (typ > 6))) or not np.isin(units,[0,2,4,8]).all()
            or not np.isin(sat,[0,1,2]).all() or np.any((sat != 0)[:,None] & (cut <= 0))):
        raise scm.InvalidAncestor("response-family")
    nonlinear = active & (typ > 0) & (typ < 6)
    piecewise = active & (typ == 6)
    if (np.any(nonlinear & (par[:,:,1] <= 0))
            or np.any(piecewise & (par[:,:,0] >= par[:,:,1]))
            or np.any(piecewise[:,:,None] & ((par[:,:,2:5] < 0) | (par[:,:,2:5] > 1)))):
        raise scm.InvalidAncestor("response-parameters")
    product = np.flatnonzero(cross)
    if (np.any(pair[product] < 0) or np.any(pair[product] >= count[product,None])
            or np.any(pair[product,0] == pair[product,1])):
        raise scm.InvalidAncestor("interaction-pair")
    # Recompute the edge Lipschitz certificate from actual solver coefficients.
    required = np.abs(coef) + np.abs(beta[:,:,None] * weights).sum(axis=1)
    required[product,pair[product,0]] += abs(cross[product])
    required[product,pair[product,1]] += abs(cross[product])
    if (not np.array_equal(budget,w.budget) or np.any(budget < 0)
            or np.any(budget[~active] != 0) or np.any(required > budget+1e-12)
            or np.any(budget.sum(axis=1) > w.rho+1e-12)):
        raise scm.InvalidAncestor("edge-or-row-contraction-budget")


def invalid_batch(job):
    split,n,family,ids = job
    rejected = []
    for candidate in ids:
        try:
            validate_world(split,n,family,candidate)
        except scm.InvalidAncestor as e:
            rejected.append((candidate,str(e)))
    return family,rejected


@dataclass(frozen=True)
class Roster:
    ids: tuple
    replaced: tuple
    rejected: tuple


def resolve(split,n,families,count,pool=None):
    """Reserve all original slots, then repair them in slot order within each family."""
    if count < 0:
        raise ValueError("ancestor count must be nonnegative")
    low,high = scm.BUCKETS[split]
    original = {f:scm.reserved_ancestors(split,n,f,count) for f in families}
    invalid = {f:{} for f in families}
    jobs = ((split,n,f,ids[j:j+32]) for f,ids in original.items() for j in range(0,count,32))
    batches = map(invalid_batch,jobs) if pool is None else pool.imap_unordered(invalid_batch,jobs)
    for family,bad in batches:
        invalid[family].update(bad)
    result = {}
    for family,ids in original.items():
        bad = invalid[family]
        effective = list(ids)
        replaced = []
        next_id = ids[-1]+1 if ids else 0
        for idx,old in enumerate(ids):
            if old not in bad:
                continue
            while True:
                if next_id > np.iinfo(np.int32).max:
                    raise OverflowError("candidate IDs exhausted the frozen plan integer type")
                candidate = next_id
                next_id += 1
                if not low <= scm.bucket(n,family,candidate) < high:
                    continue
                _,failures = invalid_batch((split,n,family,(candidate,)))
                if failures:
                    bad.update(failures)
                    continue
                effective[idx] = candidate
                replaced.append((old,candidate))
                break
        if len(set(effective)) != count:
            raise ValueError("duplicate structural ancestor after group repair")
        result[family] = Roster(tuple(effective),tuple(replaced),tuple(sorted(bad.items())))
    return result


def sidecar(path):
    return path.with_suffix(".ancestors.json")


def ledger(split,n,smoke,rosters,plan_sha256):
    return dict(version=1,profile=scm.PROFILE,split=split,width=n,smoke=smoke,plan_sha256=plan_sha256,
                families={f:dict(replaced=[list(row) for row in r.replaced],
                                 rejected=[list(row) for row in r.rejected]) for f,r in rosters.items()})


def check_plan(path,split,n,smoke,rosters):
    from .planning import PLAN_TYPE, dimensions
    count,episodes,_ = dimensions(split,smoke)
    p = np.load(path,mmap_mode="r",allow_pickle=False)
    if p.dtype != PLAN_TYPE or p.shape != (len(scm.FAMILIES)*count*episodes,):
        raise ValueError(f"frozen plan schema mismatch: {path}")
    for family_no,family in enumerate(scm.FAMILIES):
        chunk = p[family_no*count*episodes:(family_no+1)*count*episodes].reshape(count,episodes)
        ids = np.asarray(rosters[family].ids)
        if not (np.all(chunk["family"]==family_no) and np.all(chunk["ancestor"]==ids[:,None])
                and np.all(chunk["episode"]==np.arange(episodes))):
            raise ValueError(f"frozen plan disagrees with resolved ancestry in {split} N={n} {family}; "
                             "rebuild the affected width from original quotas in an isolated release")
    record = sidecar(path)
    if record.exists():
        if read_json(record) != ledger(split,n,smoke,rosters,digest(path)):
            raise ValueError(f"frozen ancestor replacement ledger or plan checksum mismatch: {record}")
    elif any(r.replaced for r in rosters.values()):
        raise ValueError(f"missing frozen ancestor replacement ledger: {record}")
    return p
