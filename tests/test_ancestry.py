"""Whole-clan repair must preserve quotas, valid frozen plans and deterministic replay."""
from dataclasses import replace
import json
import multiprocessing
import sys

import numpy as np
import pytest

from sster_data import ancestry, corpus, planning, scm
from sster_data.__main__ import main
from sster_data.replay import replay
from sster_data.storage import digest, read_json, write_json


def inject_invalid(monkeypatch, bad):
    original = scm.world
    def world(split,n,family,candidate):
        w = original(split,n,family,candidate)
        if (split,n,family,candidate) not in bad:
            return w
        rho = w.rho.copy()
        rho[0] = np.nan
        return replace(w,rho=rho)
    monkeypatch.setattr(scm,"world",world)


def replay_spawned(args):
    # Recreate the generator's initial single-thread BLAS state even if the
    # outer pytest process imported NumPy with a different BLAS thread count.
    root,row,bad = args
    with pytest.MonkeyPatch.context() as patch:
        inject_invalid(patch,{bad})
        return replay(root,row["shard"],row["id"])


def test_reserved_ids_stay_in_place_and_consecutive_invalid_spares_are_skipped(monkeypatch):
    split,n,family,count = "validation",4,"linear",7
    candidates = scm.reserved_ancestors(split,n,family,count+4)
    first = candidates[:count]
    assert ancestry.resolve(split,n,[family],count)[family].ids==first
    bad = {first[0],first[-1],*candidates[count:count+2]}
    inject_invalid(monkeypatch,{(split,n,family,i) for i in bad})
    result=ancestry.resolve(split,n,[family],count)[family]
    assert result.ids==(candidates[count+2],*first[1:-1],candidates[count+3])
    assert result.replaced==((first[0],candidates[count+2]),(first[-1],candidates[count+3]))
    assert {i for i,r in result.rejected}==bad
    assert all(r=="nonfinite-structural-parameter" for i,r in result.rejected)
    assert set(result.ids).isdisjoint(scm.ancestors("train",n,family,40))
    assert set(result.ids).isdisjoint(scm.ancestors("test",n,family,40))
    assert all(80 <= scm.bucket(n,family,i) < 90 for i in result.ids)


def test_code_errors_are_not_ancestor_or_episode_rejections(monkeypatch):
    def broken(*args):
        raise RuntimeError("program defect")
    monkeypatch.setattr(scm,"world",broken)
    with pytest.raises(RuntimeError,match="program defect"):
        ancestry.resolve("test",1,["linear"],2)


def test_well_formed_legacy_plan_opens_byte_for_byte_and_changed_group_fails_closed(tmp_path,monkeypatch):
    path=tmp_path/"plan-4.npy"
    p=planning.plan(path,"train",4,True)
    frozen=digest(path)
    del p
    ancestry.sidecar(path).unlink()
    assert len(planning.plan(path,"train",4,True))==16 and digest(path)==frozen
    bad=scm.reserved_ancestors("train",4,"linear",1)[0]
    inject_invalid(monkeypatch,{("train",4,"linear",bad)})
    with pytest.raises(ValueError,match="frozen plan disagrees"):
        planning.plan(path,"train",4,True)
    assert digest(path)==frozen and not ancestry.sidecar(path).exists()


def test_template_matching_receives_replaced_groups(monkeypatch,tmp_path):
    # Use actual matching and a small but feasible seasonal-focal cell.
    split,n,family="test",4,"linear"
    count,episodes=15,16
    bad=scm.reserved_ancestors(split,n,family,count)[0]
    replacement=scm.reserved_ancestors(split,n,family,count+1)[-1]
    inject_invalid(monkeypatch,{(split,n,family,bad)})
    monkeypatch.setattr(planning,"dimensions",lambda *args:(count,episodes,1))
    monkeypatch.setattr(planning,"quotas",lambda *args:{(n,1,"process"):4*count*episodes})
    path=tmp_path/"plan-4.npy"
    p=planning.plan(path,split,n,False)
    assert set(p[:episodes]["ancestor"])=={replacement}
    assert any(p["kind"]>=20)
    assert not np.any((p["family"]==0)&(p["ancestor"]==bad))
    from sster_data.catalog import normal_seasons
    from sster_data.episodes import schedule
    for row in p[(p["kind"]>=20)&(p["layer"]>=0)]:
        ep=planning.Episode(split,n,*planning.encode(*row))
        w=scm.world(split,n,ep.family,ep.ancestor)
        length,duration,crop=ep.clock(w.lag)
        background=normal_seasons(ep.rng("background"),n,w.warm,duration)
        assert schedule(ep,background,length,crop,w.lag,1.)[0][0].kind==ep.focus[1]


def test_generate_resume_verify_replay_and_tampered_replacement(tmp_path,monkeypatch):
    split,n,family="train",4,"linear"
    old,new=scm.reserved_ancestors(split,n,family,2)
    inject_invalid(monkeypatch,{(split,n,family,old)})
    # Admission sees the injected invalid SCM in this interpreter. The actual
    # calibration, stage and publishing processes use spawn and only see accepted IDs.
    resolver=ancestry.resolve
    monkeypatch.setattr(corpus,"resolve",lambda s,n,f,a,pool=None:resolver(s,n,f,a))
    corpus.generate(tmp_path,split,2,smoke=True)
    root=tmp_path/split
    marker=(root/"_SUCCESS").read_bytes()
    binding=ancestry.sidecar(root/"build"/f"plan-{n}.npy")
    doc=read_json(binding)
    assert doc["families"][family]["replaced"]==[[old,new]]
    audit=read_json(root/"audit.json")
    assert audit["ancestor_replacement_sha256"]=={str(n):digest(binding)}
    assert audit["windows"]==48
    rows=[json.loads(x) for x in (root/"manifest.jsonl").read_text().splitlines()]
    affected=[r for r in rows if (r["N"],r["family"],r["ancestor"])==(n,family,new)]
    assert {r["episode"] for r in affected}==set(range(4))
    assert len({r["id"] for r in rows})==48
    corpus.generate(tmp_path,split,2,smoke=True)
    monkeypatch.setattr(sys,"argv",["sster_data","verify","--out",str(tmp_path),"--split",split,"--deep"])
    main()
    assert (root/"_SUCCESS").read_bytes()==marker
    with multiprocessing.get_context("spawn").Pool(1) as workers:
        reproduced = workers.map(replay_spawned,[(root,row,(split,n,family,old)) for row in affected])
    assert reproduced == [row["id"] for row in affected]
    doc["families"][family]["replaced"][0][1]+=1
    write_json(binding,doc)
    with pytest.raises(ValueError,match="ancestor replacement ledger"):
        corpus.generate(tmp_path,split,1,smoke=True)
    with pytest.raises(ValueError,match="ancestor replacement ledger"):
        main()
    with pytest.raises(ValueError,match="ancestor replacement ledger"):
        replay(root,affected[0]["shard"],affected[0]["id"])
    assert (root/"_SUCCESS").read_bytes()==marker
    binding.unlink()
    with pytest.raises(ValueError,match="missing frozen ancestor"):
        main()


def test_plan_tamper_shared_across_all_siblings_is_detected(tmp_path):
    path=tmp_path/"plan-4.npy"
    original=planning.plan(path,"validation",4,True)
    data=original.copy()
    del original
    data["ancestor"][:4]=0  # every sibling now agrees locally, but its canonical assignment is wrong
    if np.array_equal(data["ancestor"][:4],np.load(path)["ancestor"][:4]):
        data["ancestor"][:4]=71
    np.save(path,data)
    with pytest.raises(ValueError,match="frozen plan disagrees"):
        planning.plan(path,"validation",4,True)
