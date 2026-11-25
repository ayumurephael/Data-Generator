"""Balance exported root and visible-root counts without changing physical worlds."""
from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def assign(physical, desired, rng):
    """Return physical channel at each published slot. Classes: known, other root, other."""
    shape = physical.shape
    src = np.argsort(physical + rng.random(shape), axis=1)
    dst = np.argsort(desired + rng.random(shape), axis=1)
    order = np.empty(shape, dtype=np.uint8)
    np.put_along_axis(order, dst, src, axis=1)
    return order


def balanced_orders(physical, modes, rng):
    rows, n = physical.shape
    k = (physical < 2).sum(axis=1)
    g = (physical == 0).sum(axis=1)
    keys = ((k*4 + modes)*7+g)
    order = np.empty((rows,n),dtype=np.uint8)
    leftovers = []
    for key in np.unique(keys):
        ids = rng.permutation(np.flatnonzero(keys == key))
        full = len(ids)//n*n
        leftovers.extend(ids[full:])
        for block in ids[:full].reshape(-1,n):
            # Random class base, cyclic rotations and common slot/row shuffles.
            base = physical[block[0],rng.permutation(n)]
            rolls = rng.permutation(n)
            slots = rng.permutation(n)
            desired = base[(slots[None,:]-rolls[:,None]) % n]
            order[block] = assign(physical[block],desired,rng)
    if leftovers:
        idx = np.array(leftovers)
        cls = physical[idx]
        m = len(idx)
        positions = np.arange(m*n)
        stubs = np.concatenate([rng.permutation(positions[cls.ravel() == v]) for v in range(3)])
        row = stubs // n
        box = np.repeat(np.arange(m),n)
        key = row*m+box
        by_key = np.argsort(key,kind="stable")
        unique, first, count = np.unique(key[by_key],return_index=True,return_counts=True)
        # CSR keeps multiedge multiplicities. Exhausted edges must be removed
        # because SciPy treats even explicitly stored zeros as present edges.
        perm = np.empty((m,n),dtype=np.uint8)
        for axis in rng.permutation(n):
            live = count > 0
            graph = coo_matrix((np.ones(live.sum()),(unique[live]//m,unique[live]%m)),shape=(m,m)).tocsr()
            matched = maximum_bipartite_matching(graph,perm_type="column")
            if np.any(matched < 0):
                raise ArithmeticError("regular bipartite graph has no perfect matching")
            found = np.searchsorted(unique,np.arange(m)*m+matched)
            perm[:,axis] = stubs[by_key[first[found]+count[found]-1]] % n
            count[found] -= 1
        order[idx] = perm
    raw = np.zeros(n,dtype=np.int64)
    known = raw.copy()
    for start in range(0,rows,4096):
        exported = np.take_along_axis(physical[start:start+4096],order[start:start+4096],1)
        raw += (exported < 2).sum(0)
        known += (exported == 0).sum(0)
    if np.ptp(raw) > 1 or np.ptp(known) > 1:
        raise ArithmeticError("imbalanced exported root exposure")
    return order
