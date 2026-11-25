from collections import Counter
import io
import json
import tarfile

import numpy as np
import pytest

from sster_data.balance import balanced_orders
from sster_data.catalog import MIN_LENGTH, local_wave, normal_background, normal_seasons
from sster_data.episodes import hysteresis, observe, realize, schedule
from sster_data.planning import Episode, FAMILIES, match_seasons, quotas
from sster_data.scm import ancestors, solve, world
from sster_data.storage import Archive,pack,unpack


def test_full_focal_catalogue_on_frozen_backgrounds():
    from sster_data.planning import cover, PLAN_TYPE, encode, templates
    records = np.zeros(320,dtype=PLAN_TYPE)
    records["ancestor"] = np.arange(320)
    records["k"],records["mode"] = 1,1
    records["layer"] = records["kind"] = records["severity"] = -1
    cover(records,np.arange(len(records)),"train",1,"process",1)
    obtained = set()
    for row in records[records["layer"] >= 0]:
        ep = Episode("train",1,*encode(*row))
        wld = world("train",1,ep.family,ep.ancestor)
        w,length,crop = ep.clock(wld.lag)
        background = normal_seasons(ep.rng("background"),1,wld.warm,length)
        events,_ = schedule(ep,background,w,crop,wld.lag,1.)
        assert events[0].kind == ep.focus[1] and events[0].layer == ep.focus[0]
        assert all(crop <= e.start < e.stop <= crop+w for e in events)
        obtained.add(ep.focus[:2])
    assert obtained == set(templates("process"))


def test_physically_active_but_unobservable_root_is_never_negative():
    ep = Episode("test",4,"linear",1,0,1,"process",None,True)
    normal = np.zeros((600,4))
    direct = np.zeros_like(normal,dtype=bool)
    direct[200:211,2] = True
    y = direct.any(1)
    _,tgt,meta = observe(ep,256,600,150,normal,normal.copy(),np.zeros_like(normal),direct,1.,y)
    assert tgt["R"].tolist() == [False,False,True,False]
    assert not tgt["R_known"].any() and not tgt["Y_known"][tgt["Y"]].any()
    assert meta["no_visible_root"] and meta["no_time_witness"]


@pytest.mark.parametrize("split,windows", [("train",6720000),("validation",336000),("test",336000)])
def test_exact_cell_schedule(split,windows):
    cells = quotas(split)
    totals = Counter()
    by_k = Counter()
    for (n,k,mode),v in cells.items():
        assert k <= n
        totals[mode] += v
        if k:
            by_k[k] += v
    assert sum(cells.values()) == windows
    assert list(totals[k] for k in ("healthy","process","observation","mixed")) == [
        windows*30//100,windows*35//100,windows*28//100,windows*7//100]
    assert len(set(by_k.values())) == 1


def test_visible_root_exposure_remainders_and_parallel_edges():
    rng = np.random.default_rng(235)
    for n in (1,4,8,16,32):
        count = 201
        role = np.full((count,n),2,dtype=np.uint8)
        for row in role:
            k = int(rng.integers(min(n,6)+1))
            sites = rng.choice(n,k,replace=False)
            row[sites] = 1
            row[sites[:rng.integers(k+1)]] = 0
        perm = balanced_orders(role,rng.integers(0,4,count),rng)
        assert np.all(np.sort(perm,axis=1) == np.arange(n))
        exported = np.take_along_axis(role,perm,axis=1)
        assert np.ptp((exported<2).sum(axis=0)) <= 1
        assert np.ptp((exported==0).sum(axis=0)) <= 1


def test_matching_reassigns_scarce_tickets():
    feasible = np.array([[True,True],[True,False]])
    assert sorted(match_seasons(feasible,1)) == [(0,1),(1,0)]
    assert match_seasons(feasible,2) is None


@pytest.mark.parametrize("k",FAMILIES)
def test_paired_solver_and_budget(k):
    w = world("train",8,k,2)
    rng = np.random.default_rng(42)
    drives = rng.normal(size=(128,8))
    shocks = np.zeros_like(drives)
    shocks[33:50,3] = 3.
    x,xf,u = solve(drives,shocks,True,*w.args)
    assert np.array_equal(x[:33],xf[:33])
    assert np.all(abs(x-xf) <= u+1e-12)
    assert np.all(w.budget.sum(1)<=w.rho+1e-12)
    single = solve(drives+shocks,np.empty((0,0)),False,*w.args)[0]
    np.testing.assert_allclose(single,xf,rtol=1e-13,atol=1e-13)


def test_lagged_chain_and_hysteresis():
    w = world("train",3,"linear",0)
    args = [x.copy() for x in w.args]
    count,parent,delay,typ,coeff,par,pair,cross,units,weights,bias,beta,tbias,sat,cut,budget = args
    for arr in args:
        arr[...] = 0
    count[:] = [0,1,1]
    parent[1,0],parent[2,0] = 0,1
    delay[1,0],delay[2,0] = 2,1
    coeff[1,0],coeff[2,0] = .5,-.4
    budget[:] = abs(coeff)
    drives = np.zeros((20,3))
    j = drives.copy()
    j[10,0] = 2.
    x,xf,u = solve(drives,j,True,*args)
    assert xf[12,1] == 1 and xf[13,2] == -.4
    np.testing.assert_allclose(abs(xf-x),u)
    y,open_end = hysteresis(abs(xf-x).max(axis=1))
    assert np.flatnonzero(y).tolist() == [10,11,12,13]
    assert not open_end
    assert hysteresis(np.array([.5,0.,0.]))[0].all()


def test_observation_only_and_replay():
    ep = Episode("train",4,"saturating",1,9,2,"observation",("observation","L07",1),True)
    inp,tgt,meta,paths = realize(ep,1.,audit=True)
    assert np.array_equal(paths["normal"],paths["factual"])
    assert np.array_equal(paths["y"],paths["direct"].any(axis=1))
    assert tgt["R"].sum() == 2
    again = realize(ep,1.)[0]
    for key in inp:
        assert np.array_equal(inp[key],again[key])
    assert meta["events"][0]["kind"] == "L07"
    assert meta["events"][0]["stop"]-meta["events"][0]["start"] == 1


def test_background_extension_and_all_local_operators():
    n,warm,t = 4,64,259
    short = normal_background(np.random.default_rng(1),n,warm,t)
    extended = normal_background(np.random.default_rng(1),n,warm,t,extra=17)
    np.testing.assert_array_equal(short.b,extended.b[:len(short.b)])
    light = normal_seasons(np.random.default_rng(1),n,warm,t)
    for i in range(n):
        assert set(short.eligible(i,100,160)) == set(light.eligible(i,100,160))
    for kind,minimum in MIN_LENGTH.items():
        if kind[0]=="L":
            arr = local_wave(kind,minimum,np.random.default_rng(1),2.)
            assert arr.shape == (minimum,) and np.isfinite(arr).all() and abs(arr).max() > 0


def test_split_and_random_access(tmp_path):
    a,b,c = [set(ancestors(split,4,"additive",100)) for split in ("train","validation","test")]
    assert not (a&b or b&c or c&a)
    path = tmp_path/"data.tar"
    with Archive(path) as tar:
        h1,s1 = tar.add("one.npz",pack({"test":np.arange(24)}))
        h2,s2 = tar.add("two.npz",pack({"test":np.arange(3)}))
    with open(path,"rb") as f:
        for pos,size,expected in ((h1,s1,24),(h2,s2,3)):
            f.seek(pos)
            assert unpack(f.read(size))["test"].tolist() == list(range(expected))
