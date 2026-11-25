"""Numerical rejection must not change matched seasonal parameters or the fault ledger."""
import json
import tarfile

import numpy as np
import pytest

from sster_data import catalog, episodes
from sster_data.corpus import paths, prepare_orders, publish_batch, stage
from sster_data.diagnostics import metadata_kinds
from sster_data.planning import Episode, PLAN_TYPE, encode
from sster_data.replay import replay
from sster_data.scm import world
from sster_data.storage import read_json, write_json


# Real production validation plan, N=1, row 36386, from the original quota plan.
ROW = (3,213,2,1,1,0,36,-1)
SCALE = 1.5027120088400179


def failure_episode():
    row = np.array([ROW],dtype=PLAN_TYPE)[0]
    return Episode("validation",1,*encode(*row))


def test_exact_production_failure_and_repaired_event():
    ep = failure_episode()
    assert ep.id == "0c1b453808b2bb3cc882b53bf970463a"
    wld = world(ep.split,ep.n,ep.family,ep.ancestor)
    w,length,crop = ep.clock(wld.lag)
    original = catalog.normal_seasons(ep.rng("background"),1,wld.warm,length)
    legacy_retry = catalog.normal_seasons(ep.rng("background",1),1,wld.warm,length)
    assert original.components[0]["kind"] == "wavelet"
    assert legacy_retry.components[0]["kind"] == "sine"
    assert "S17" not in legacy_retry.eligible(0,crop+64,crop+w-64)
    with pytest.raises(episodes.NumericalFailure,match="flat"):
        episodes.realize(ep,SCALE)
    public,targets,meta,traces = episodes.realize(ep,SCALE,retry=1,audit=True)
    assert meta["season_retry"] == 0 and meta["retry"] == 1
    assert meta["events"][0]["kind"] == "S17"
    assert (meta["events"][0]["start"],meta["events"][0]["stop"]) == (1579,1661)
    assert traces["normal"].std()>0.001
    assert targets["R"].tolist() == [True] and not targets["R_known"].any()
    assert int(targets["Y"].sum()) == 82
    assert metadata_kinds(ep,1)[:,1].tolist() == ["wavelet"]


@pytest.mark.parametrize("mode,k", [("healthy",0),("process",3),("observation",2),("mixed",4)])
def test_every_mode_keeps_whole_normal_season_and_clock(mode,k):
    ep = Episode("validation",8,"additive",19,7,k,mode,None)
    wld = world(ep.split,ep.n,ep.family,ep.ancestor)
    w,ordinary,crop = ep.clock(wld.lag)
    frozen = catalog.normal_seasons(ep.rng("background"),ep.n,wld.warm,ordinary)
    baseline = catalog.normal_background(ep.rng("background"),ep.n,wld.warm,ordinary)
    retried = catalog.normal_background(ep.rng("background",2),ep.n,wld.warm,ordinary,frozen_seasons=frozen)
    assert baseline.components == retried.components
    np.testing.assert_array_equal(baseline.seasonal,retried.seasonal)
    assert not np.array_equal(baseline.b,retried.b)
    assert episodes.schedule(ep,baseline,w,crop,wld.lag,1.) == episodes.schedule(ep,retried,w,crop,wld.lag,1.)
    assert metadata_kinds(ep,2)[:,1].tolist() == [c["kind"] for c in frozen.components]


def test_realize_guard_reuses_retry_and_original_season(monkeypatch):
    ep = Episode("test",4,"linear",9,0,1,"observation",None)
    def end_crop(self,lag):
        return 256,600,344
    def final_tick(ep,bg,w,crop,lag,scale):
        return [episodes.Event("0",0,"observation",2,"L07",599,600,scale,"medium",
                               {"stream":"event/0/0/operator"})],"single"
    monkeypatch.setattr(Episode,"clock",end_crop)
    monkeypatch.setattr(episodes,"schedule",final_tick)
    backgrounds = []
    original = catalog.normal_background
    def record(*args,**kwargs):
        background = original(*args,**kwargs)
        backgrounds.append(background)
        return background
    monkeypatch.setattr(catalog,"normal_background",record)
    _,targets,meta,_ = episodes.realize(ep,1.,retry=1)
    assert meta["extra_guard"]>=3 and meta["season_retry"]==0 and targets["R"].sum()==1
    ordinary,extended=backgrounds
    assert extended.components==ordinary.components
    np.testing.assert_array_equal(ordinary.b,extended.b[:len(ordinary.b)])
    np.testing.assert_array_equal(ordinary.seasonal,extended.seasonal[:len(ordinary.b)])


def test_production_stage_publish_replay_and_legacy_receipt_guard(tmp_path):
    ep = failure_episode()
    root = tmp_path/"validation"
    for name in ("build","shards","private","receipts"):
        (root/name).mkdir(parents=True)
    p = np.array([ROW],dtype=PLAN_TYPE)
    _,plan,_,_,receipt = paths(root,1,0)
    np.save(plan,p)
    job=(root,"validation",False,1,0,1,{ep.family:SCALE})
    stage(job)
    prepare_orders(root,1,p,[0])
    publish_batch((root,1,0))
    rc = read_json(receipt)
    assert rc["rows"][0]["rejections"]==1 and rc["rows"][0]["season_retry"]==0
    with tarfile.open(root/rc["private"],"r:") as tar:
        m=json.load(tar.extractfile(ep.id+".replay.json"))
        assert m["rejections"] == ["flat"] and m["season_retry"]==0
    assert replay(root,rc["public"],ep.id)==ep.id
    stage(job)  # committed current receipt can be resumed
    del rc["rows"][0]["season_retry"]
    write_json(receipt,rc)
    with pytest.raises(ValueError,match="legacy numerical retry"):
        stage(job)
