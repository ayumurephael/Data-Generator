"""Stage an edition's unchanged tickets from the parent corpus in their original physical axes."""
from collections import Counter
from contextlib import ExitStack
from functools import lru_cache
from itertools import groupby
from pathlib import Path
import json
import tarfile

import numpy as np

from . import corpus
from .episodes import require_frozen_seasons
from .planning import Episode, encode as decode_record
from .scm import PROFILE
from .storage import Archive, digest, encode, pack, read_json, unpack, write_json


@lru_cache(maxsize=128)
def _hash(path,size,mtime_ns):
    return digest(path)


def _checksum(path):
    stat=path.stat()
    return _hash(path,stat.st_size,stat.st_mtime_ns)


@lru_cache(maxsize=16)
def _closed_table(path,checksum):
    sealed=read_json(path)
    rows={r["begin"]:r for r in sealed["shards"]}
    if len(rows)!=len(sealed["shards"]):
        raise ValueError("duplicate staged shard in frozen spool")
    return sealed["identity"],rows


def _ticket(meta,ep,scale):
    if (meta["id"],meta["split"],meta["family"],meta["ancestor"],meta["episode"],meta["N"],meta["K"],
            meta["mode"],meta["focal"],meta["scale"],meta["profile"],meta["smoke"]) != (
            ep.id,ep.split,ep.family,ep.ancestor,ep.episode,ep.n,ep.k,ep.mode,list(ep.focus) if ep.focus else None,
            scale,PROFILE,ep.smoke):
        raise ValueError(f"parent envelope differs from selected ticket {ep.id}")
    require_frozen_seasons(meta,meta["retry"])


def _payload(handle,row,part,uid):
    offset=row[part+"_offset"]
    handle.seek(offset-512)
    header=tarfile.TarInfo.frombuf(handle.read(512),"utf8","strict")
    if header.name!=uid+"."+part+"s.npz" or header.size!=row[part+"_size"]:
        raise ValueError("public member name or offset differs from parent receipt")
    data=handle.read(header.size)
    if len(data)!=header.size:
        raise IOError("short public data member")
    return unpack(data)


def _physical_from_public(archive,src,ep,scale,index,audit):
    pub,private,receipt=src
    row=receipt["rows"][index]
    if row["id"]!=ep.id:
        raise ValueError("parent public receipt has a different episode ID")
    meta=json.loads(corpus.read_member(private,ep.id+".replay.json"))
    _ticket(meta,ep,scale)
    trace=meta.pop("physical_trace",None)
    if audit and trace is None:
        return None
    inp=_payload(pub,row,"input",ep.id)
    tgt=_payload(pub,row,"target",ep.id)
    corpus.validate_arrays(inp,tgt,meta)
    inverse=np.asarray(meta.pop("physical_to_exported"))
    if inverse.shape!=(ep.n,) or not np.issubdtype(inverse.dtype,np.integer) or not np.array_equal(np.sort(inverse),np.arange(ep.n)):
        raise ValueError("invalid physical-to-exported permutation")
    meta.pop("scale_reference")
    for event in meta["events"]:
        if event.pop("sensor")!=inverse[event["target"]]:
            raise ValueError("event sensor disagrees with physical target")
    # inverse[i] is the exported column containing physical sensor i.
    arrays={k:(v if k in ("time","Y","Y_known") else np.take(v,inverse,axis=-1))
            for k,v in {**inp,**tgt}.items()}
    roots=np.flatnonzero(arrays["R"]).tolist()
    known=np.flatnonzero(arrays["R"] & arrays["R_known"]).tolist()
    archive.add(ep.id+".npz",pack(arrays))
    archive.add(ep.id+".json",encode(meta))
    if audit:
        if trace!=ep.id+".trace.npz":
            raise ValueError("invalid physical trace name")
        archive.add(trace,corpus.read_member(private,trace))
    return [ep.id,roots,known]


def _physical_from_stage(archive,src,ep,scale,index,audit):
    tar,roles=src
    row=roles["roles"][index]
    if row[0]!=ep.id:
        raise ValueError("parent physical stage has a different episode ID")
    meta_bytes=corpus.read_member(tar,ep.id+".json")
    _ticket(json.loads(meta_bytes),ep,scale)
    trace_name=ep.id+".trace.npz"
    if audit:
        try:
            trace=tar.getmember(trace_name)
        except KeyError:
            return None
    archive.add(ep.id+".npz",corpus.read_member(tar,ep.id+".npz"))
    archive.add(ep.id+".json",meta_bytes)
    if audit:
        archive.add(trace_name,tar.extractfile(trace).read())
    return row


def _source(stack,parent,extras,n,begin,bank_hash,plan_hash):
    tag,_,stage,roles,receipt=corpus.paths(parent,n,begin)
    if receipt.exists():
        r=read_json(receipt)
        if (r["public"],r["private"])!=(f"shards/{tag}.tar",f"private/{tag}.tar"):
            raise ValueError("parent receipt contains incorrect archive paths")
        for kind in ("public","private"):
            if _checksum(parent/r[kind])!=r[kind+"_sha256"]:
                raise IOError("damaged parent "+kind+" archive")
        public=stack.enter_context((parent/r["public"]).open("rb"))
        private=stack.enter_context(tarfile.open(parent/r["private"],"r:"))
        return "published",(public,private,r)
    locations=[(stage,roles,None)]
    for build in extras:
        if n!=128:
            continue
        seal=build.parent.parent/"CLOSED.json"
        identity,rows=_closed_table(seal,_checksum(seal))
        if identity["plan"]!=plan_hash or identity["calibration"]!=bank_hash:
            raise ValueError("sealed staging spool belongs to another parent plan or calibration")
        if begin in rows:
            locations.append((build/(tag+".stage.tar"),build/(tag+".roles.json"),rows[begin]))
    for tar_path,roles_path,proof in locations:
        if not roles_path.exists() and proof is None:
            continue
        roles_data=read_json(roles_path)
        if not tar_path.is_file() or _checksum(tar_path)!=roles_data["sha256"]:
            raise IOError("missing or damaged committed physical stage")
        if proof and (proof["roles"]!=_checksum(roles_path) or proof["physical"]!=roles_data["sha256"]):
            raise IOError("sealed physical spool commitment mismatch")
        tar=stack.enter_context(tarfile.open(tar_path,"r:"))
        return ("spool" if proof else "physical"),(tar,roles_data)
    return "missing",None


def stage(job,*,source_root,extra_builds=(),require_material=False):
    """Perform a native 64-row staging job; source_root is the read-only parent split."""
    root,split,smoke,n,begin,end,scales=job
    root,parent=Path(root),Path(source_root)
    if root.resolve()==parent.resolve():
        raise ValueError("new edition must use an independent destination")
    tag,path,target,roles_path,receipt=corpus.paths(root,n,begin)
    summary=root/"build"/(tag+".reuse.json")
    if receipt.exists() or roles_path.exists() and target.exists():
        corpus.stage(job)
        if not summary.exists():
            totals={"verified_existing":end-begin}
            write_json(summary,totals)
            return totals
        totals=read_json(summary)
        allowed={"published","physical","spool","missing","native_for_trace","verified_existing"}
        if (not isinstance(totals,dict) or not set(totals)<=allowed
                or any(type(v) is not int or v<0 for v in totals.values()) or sum(totals.values())!=end-begin):
            raise ValueError("reused chunk provenance is incomplete")
        return totals
    parent_path=parent/"build"/f"plan-{n}.npy"
    index_path=root/"build"/f"source-rows-{n}.npy"
    p=corpus.read_plan(str(path))
    old=corpus.read_plan(str(parent_path))
    index=corpus.read_plan(str(index_path))
    candidates=index[begin:end]
    if (index.dtype!=np.dtype("<i4") or index.shape!=p.shape or end>len(p) or begin>=end
            or (candidates<0).any() or (candidates>=len(old)).any() or (np.diff(candidates)<=0).any()
            or not np.array_equal(p[begin:end],old[candidates])):
        raise ValueError("selected parent row map differs from the frozen tickets")
    bank_hash=_checksum(parent.parent/"calibration.json")
    if _checksum(root.parent/"calibration.json")!=bank_hash:
        raise ValueError("selected edition has another calibration bank")
    plan_hash=_checksum(parent_path)
    extras=tuple(Path(b) for b in extra_builds)
    counts=Counter()
    roles=[]
    with Archive(target) as archive:
        for source_begin,items in groupby(enumerate(candidates,begin),lambda x:int(x[1])//corpus.BATCH*corpus.BATCH):
            with ExitStack() as stack:
                origin,src=_source(stack,parent,extras,n,source_begin,bank_hash,plan_hash)
                for serial,old_serial in items:
                    ep=Episode(split,n,*decode_record(*p[serial]),smoke=smoke)
                    audit=serial%8192==0
                    label=origin
                    row=None
                    if origin!="missing":
                        reader=_physical_from_public if origin=="published" else _physical_from_stage
                        row=reader(archive,src,ep,scales[ep.family],int(old_serial)-source_begin,audit)
                        if row is None:
                            label="native_for_trace"
                    if row is None:
                        if require_material:
                            raise ValueError(f"GPU-native work was incorrectly assigned to a material-only worker: {ep.id}")
                        row=corpus.write_native_episode(archive,ep,scales[ep.family],audit)
                    counts[label]+=1
                    roles.append(row)
    write_json(summary,dict(counts))
    write_json(roles_path,dict(sha256=digest(target),roles=roles))
    return dict(counts)
