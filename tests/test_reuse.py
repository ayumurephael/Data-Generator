"""Recover exact pre-export bytes while subsampling an independently rebalanced edition."""
import json

import numpy as np
import pytest

from sster_data import corpus,planning,reuse
from sster_data.storage import digest,read_json,write_json

def setup(root,p,n):
    for name in ("build","private","shards","receipts"):
        (root/name).mkdir(parents=True)
    write_json(root.parent/"calibration.json",dict(scales={f:1.125 for f in planning.FAMILIES}))
    np.save(root/"build"/f"plan-{n}.npy",p)


def rows(count):
    a=np.zeros(count,dtype=planning.PLAN_TYPE)
    for i in range(count):
        k=i%3
        a[i]=(i%4,8,i,k,(0 if not k else 1 if k==1 else 3),-1,-1,-1)
    return a


@pytest.mark.parametrize("first,mode",[(0,"published"),(1,"native_for_trace")])
def test_mixed_sources_match_direct_native_archives(tmp_path,first,mode):
    n=4
    parent=tmp_path/"old/train"
    new=tmp_path/"edition/train"
    direct=tmp_path/"regenerated/train"
    p=rows(132)
    ix=np.array([first,2,3,65,66,129,130],dtype="<i4")
    scale={f:1.125 for f in planning.FAMILIES}
    for root,plan in ((parent,p),(new,p[ix]),(direct,p[ix])):
        setup(root,plan,n)
    np.save(new/"build"/f"source-rows-{n}.npy",ix)
    # Give the parent a non-self-inverse axis order to catch either permutation direction.
    np.save(parent/"build"/f"order-{n}.npy",np.tile([2,0,3,1],(len(p),1)).astype(np.uint8))
    for start in (0,64):
        corpus.stage((parent,"train",True,n,start,start+4,scale))
    corpus.publish_batch((parent,n,0))
    job=(new,"train",True,n,0,len(ix),scale)
    counts=reuse.stage(job,source_root=parent)
    assert counts.get(mode)>=1 and counts["physical"]==2 and counts["missing"]==2
    assert counts["published"]==3-int(first==1)
    assert reuse.stage(job,source_root=parent)==counts
    corpus.stage((direct,"train",True,n,0,len(ix),scale))
    for i in (2,3):
        assert corpus.paths(new,n,0)[i].read_bytes()==corpus.paths(direct,n,0)[i].read_bytes()
    for base in (new,direct):
        corpus.prepare_orders(base,n,p[ix],[0])
        corpus.publish_batch((base,n,0))
    for folder in ("shards","private"):
        path=f"{folder}/n{n:03}-0000000.tar"
        assert (new/path).read_bytes()==(direct/path).read_bytes()
    assert reuse.stage(job,source_root=parent)==counts


def test_rejects_changed_parent_ticket_and_damaged_public_archive(tmp_path):
    n=4; p=rows(4);ix=np.array([0,2],dtype="<i4")
    src=tmp_path/"src/train";dest=tmp_path/"dest/train"
    setup(src,p,n);setup(dest,p[ix],n)
    index=dest/"build"/f"source-rows-{n}.npy"
    np.save(index,np.array([0,3],dtype="<i4"))
    scale={f:1.125 for f in planning.FAMILIES}
    job=(dest,"train",True,n,0,2,scale)
    with pytest.raises(ValueError,match="row map"):
        reuse.stage(job,source_root=src)
    corpus.read_plan.cache_clear()
    index.unlink();np.save(index,ix)
    corpus.stage((src,"train",True,n,0,4,scale))
    corpus.prepare_orders(src,n,p,[0]);corpus.publish_batch((src,n,0))
    tar=next((src/"shards").glob("*.tar"))
    with tar.open("r+b") as f:f.write(b"!")
    with pytest.raises(IOError,match="damaged parent"):
        reuse.stage(job,source_root=src)


def test_sealed_extra_spool_matches_native_and_detects_hash_damage(tmp_path):
    n=128;p=rows(2);ix=np.array([0,1],dtype="<i4")
    parent=tmp_path/"r2/train";extra=tmp_path/"spool/train";dest=tmp_path/"current/train"
    for root in (parent,extra,dest):
        setup(root,p,n)
    np.save(dest/"build"/"source-rows-128.npy",ix)
    scale={f:1.125 for f in planning.FAMILIES}
    corpus.stage((extra,"train",True,n,0,2,scale))
    _,_,tar,roles,_=corpus.paths(extra,n,0)
    info=dict(begin=0,physical=digest(tar),roles=digest(roles))
    proof=dict(shards=[info],identity=dict(plan=digest(parent/"build"/"plan-128.npy"),
                                          calibration=digest(parent.parent/"calibration.json")))
    seal=extra.parent/"CLOSED.json"
    write_json(seal,proof)
    job=(dest,"train",True,n,0,2,scale)
    assert reuse.stage(job,source_root=parent,extra_builds=(extra/"build",))=={"spool":2}
    assert corpus.paths(dest,n,0)[2].read_bytes()==tar.read_bytes()
    assert corpus.paths(dest,n,0)[3].read_bytes()==roles.read_bytes()
    other=tmp_path/"tamper/train";setup(other,p,n)
    np.save(other/"build"/"source-rows-128.npy",ix)
    proof["shards"][0]["roles"]="0"*64;write_json(seal,proof)
    with pytest.raises(IOError,match="commitment"):
        reuse.stage((other,*job[1:]),source_root=parent,extra_builds=(extra/"build",))


def test_native_chunk_resumes_without_claiming_a_source(tmp_path):
    n=4;p=rows(2);dest=tmp_path/"current/train";parent=tmp_path/"old/train"
    setup(dest,p,n);setup(parent,p,n)
    job=(dest,"train",True,n,0,2,{f:1.125 for f in planning.FAMILIES})
    corpus.stage(job)
    counts=reuse.stage(job,source_root=parent)
    assert counts=={"verified_existing":2}
    provenance=dest/"build"/"n004-0000000.reuse.json"
    write_json(provenance,{"invalid":2})
    with pytest.raises(ValueError,match="provenance"):
        reuse.stage(job,source_root=parent)
