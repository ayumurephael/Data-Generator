"""Resumable, deterministic width-at-a-time corpus builder."""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import tarfile

import numpy as np

from .balance import balanced_orders
from .calibration import calibration
from .episodes import NumericalFailure, realize, require_frozen_seasons
from .planning import dimensions, encode as decode_record, Episode, MODES, plan, quotas, templates, widths
from .scm import FAMILIES, PROFILE, ancestors, bucket, rng_for
from .storage import Archive, atomic, digest, encode, lock, pack, read_json, unpack, write_json

BATCH = 64
INPUTS = ("values","observed","time","channel_valid")
TARGETS = ("Y","Y_known","R","R_known","E")
SHARD_LIMIT = 256*1024**2


@lru_cache(maxsize=8)
def read_plan(path):
    return np.load(path,mmap_mode="r",allow_pickle=False)


def paths(root, n, begin):
    tag = f"n{n:03d}-{begin:07d}"
    return tag,root/"build"/f"plan-{n}.npy",root/"build"/(tag+".stage.tar"),root/"build"/(tag+".roles.json"),root/"receipts"/(tag+".json")


def read_member(tar, name):
    member = tar.extractfile(name)
    if member is None:
        raise IOError(f"missing archive member {name}")
    return member.read()


def stage(job):
    root,split,smoke,n,begin,end,scales = job
    tag,plan_path,stage_path,roles_path,receipt = paths(root,n,begin)
    if receipt.exists():
        if not roles_path.exists():
            raise IOError("committed shard missing its exposure receipt")
        for row in read_json(receipt)["rows"]:
            require_frozen_seasons(row,row["rejections"])
        return
    if roles_path.exists() and stage_path.exists():
        roles = read_json(roles_path)
        if digest(stage_path) == roles["sha256"]:
            return
        raise IOError(f"damaged physical staging archive {tag}")
    p = read_plan(str(plan_path))
    roles = []
    with Archive(stage_path) as archive:
        for serial in range(begin,end):
            ep = Episode(split,n,*decode_record(*p[serial]),smoke=smoke)
            failures = []
            for retry in range(50):
                try:
                    inp,tgt,meta,traces = realize(ep,scales[ep.family],retry,audit=(serial%8192==0))
                    break
                except (NumericalFailure, np.linalg.LinAlgError) as err:
                    failures.append(str(err))
            else:
                raise RuntimeError(f"numerical rejection exhausted for {ep.id}: {failures}")
            meta.update(rejections=failures,profile=PROFILE,smoke=smoke)
            archive.add(ep.id+".npz",pack({**inp, **tgt}))
            archive.add(ep.id+".json",encode(meta))
            if traces is not None:
                archive.add(ep.id+".trace.npz",pack(traces))
            known = np.flatnonzero(tgt["R"] & tgt["R_known"]).tolist()
            roots = np.flatnonzero(tgt["R"]).tolist()
            roles.append([ep.id,roots,known])
    write_json(roles_path,dict(sha256=digest(stage_path),roles=roles))


def validate_arrays(inp,tgt,meta):
    n,t,w,crop = meta["N"],meta["T"],meta["w"],meta["crop_start"]
    sizes = dict(values=(t,n),observed=(t,n),time=(t,),channel_valid=(n,),Y=(t,),Y_known=(t,),
                 R=(n,),R_known=(n,),E=(n,))
    all_arrays = {**inp,**tgt}
    if set(inp) != set(INPUTS) or set(tgt) != set(TARGETS):
        raise ValueError("invalid key allowlist")
    for key,shape in sizes.items():
        a = all_arrays[key]
        dtype = {"values":"float32","time":"float64"}.get(key,"bool")
        if a.shape != shape or a.dtype != dtype:
            raise ValueError(f"invalid {key} shape/dtype")
    if not np.isfinite(inp["values"]).all() or np.any(inp["values"][~inp["observed"]]):
        raise ValueError("nonfinite or unmasked public values")
    if not (np.diff(inp["time"]) > 0).all() or tuple(inp["time"][[0,-1]]) != (crop,crop+w-1):
        raise ValueError("invalid crop clock")
    if not np.array_equal(tgt["E"],inp["channel_valid"]) or (tgt["R_known"] & ~tgt["E"]).any():
        raise ValueError("invalid candidate mask")
    if np.any(~tgt["Y"] & ~tgt["Y_known"]):
        raise ValueError("unknown healthy time labels")
    if not tgt["R"].any() and (tgt["Y"].any() or not tgt["R_known"].all()):
        raise ValueError("invalid clean supervision")
    if tgt["R"].any() and not (tgt["R"] & tgt["R_known"]).any() and tgt["R_known"].any():
        raise ValueError("unwitnessed injected window supervised as negative")
    events = meta["events"]
    if set(ev["sensor"] for ev in events) != set(np.flatnonzero(tgt["R"])):
        raise ValueError("root labels disagree with direct event ledger")
    if len(set(ev["target"] for ev in events)) != meta["K"] or len(events) > 2*meta["K"]:
        raise ValueError("source count differs from physical event ledger")
    for ev in events:
        if not crop <= ev["start"] < ev["stop"] <= crop+w or not (ev["stop"] <= meta["T_grid"]):
            raise ValueError("invalid event interval")


def sidecar(meta,inp,tgt,public,private,checksum):
    uid = meta["id"]
    def arrays(mapping,group):
        return {k:dict(file=public,entry=f"{uid}.{group}.npz",key=k,dtype=str(v.dtype),shape=list(v.shape))
                for k,v in mapping.items()}
    return dict(protocol_version=PROFILE,sample_id=uid,
                ancestor=dict(family_id=meta["family"],family_seed=meta["ancestor"],split=meta["split"]),
                dimensions={**{k:meta[k] for k in ("T","N","L","T_grid","crop_start","w")},"V":meta["N"],"dt":1},
                arrays=dict(inputs=arrays(inp,"inputs"),targets=arrays(tgt,"targets")),events=meta["events"],
                policy_ref="sster_data/config/profile.json",calibration_ref="../calibration.json",
                private_replay=dict(path=f"{private}#{uid}.replay.json",sha256=checksum),
                audit=dict(right_censored=meta["right_censored"],outside_candidate_cause=False,annotation_unknown=False,
                           source_groups=meta["K"],root_sensor_count=meta["K"],input_allowlist=list(INPUTS),mode=meta["mode"]))


def publish_batch(job):
    root,n,begin = job
    tag,_,src,roles_path,receipt = paths(root,n,begin)
    if receipt.exists():
        # The manifest finalizer checks file hashes before release.
        if src.exists():
            src.unlink()
        return
    roles = read_json(roles_path)["roles"]
    orders = read_plan(str(root/"build"/f"order-{n}.npy"))
    pubname, privname = f"shards/{tag}.tar",f"private/{tag}.tar"
    rows = []
    with tarfile.open(src,"r:") as old, Archive(root/pubname) as public, Archive(root/privname) as private:
        for local,(uid,_,_) in enumerate(roles):
            serial = begin+local
            order = orders[serial]
            inverse = np.argsort(order)
            data = unpack(read_member(old,uid+".npz"))
            meta = json.loads(read_member(old,uid+".json"))
            require_frozen_seasons(meta,meta["retry"])
            meta["physical_to_exported"] = inverse.tolist()
            meta["scale_reference"] = "../calibration.json"
            for ev in meta["events"]:
                ev["sensor"] = int(inverse[ev["target"]])
            inp = {k:data[k] if k == "time" else np.take(data[k],order,axis=-1) for k in INPUTS}
            tgt = {k:data[k] if k in ("Y","Y_known") else data[k][order] for k in TARGETS}
            validate_arrays(inp,tgt,meta)
            if serial%8192 == 0:
                name = uid+".trace.npz"
                private.add(name,read_member(old,name))
                meta["physical_trace"] = name
            payload = encode(meta)
            private.add(uid+".replay.json",payload)
            index = sidecar(meta,inp,tgt,pubname,privname,hashlib.sha256(payload).hexdigest())
            private.add(uid+".index.json",encode(index))
            istart,isize = public.add(uid+".inputs.npz",pack(inp))
            tstart,tsize = public.add(uid+".targets.npz",pack(tgt))
            row = {k:meta[k] for k in ("id","ancestor","episode","family","N","K","mode","T","crop_start","w","focal")}
            raw = np.flatnonzero(tgt["R"])
            known = np.flatnonzero(tgt["R"] & tgt["R_known"])
            row.update(shard=pubname,input_offset=istart,input_size=isize,target_offset=tstart,target_size=tsize,
                       raw=raw.tolist(),known=known.tolist(),
                       events=[[e["layer"],e["kind"],witness,e["severity"]] for e,witness in zip(meta["events"],meta["event_witness"])],
                       quality={k:meta[k] for k in ("no_visible_root","only_propagation_visible","no_time_witness",
                           "all_positive_rows_thinned","new_propagation_onset_after_last_event","right_censored")},
                       layout=meta["layout"], acquisition=meta["acquisition"], rejections=len(meta["rejections"]))
            positives = inp["time"][tgt["Y"]]
            anchor_rows = positives if len(positives) else inp["time"]
            row["anchor"] = int(anchor_rows[len(anchor_rows)//2])
            row["focal_witness"] = bool(meta["focal"] and meta["event_witness"][0])
            if meta["retry"]:
                row["season_retry"] = meta["season_retry"]
            rows.append(row)
    # 64 rows * maximum public tuple < 90 MiB even before compression.
    # The byte check also guards against future changes in the maximum shape.
    if (root/pubname).stat().st_size > SHARD_LIMIT and len(rows) > 1:
        raise ValueError("archive exceeds 256 MiB")
    write_json(receipt,dict(rows=rows,public=pubname,private=privname,public_sha256=digest(root/pubname),
                            private_sha256=digest(root/privname)))
    src.unlink()


def prepare_orders(root,n,p,starts):
    path = root/"build"/f"order-{n}.npy"
    if path.exists():
        return
    physical = np.full((len(p),n),2,dtype=np.uint8)
    for begin in starts:
        roles = read_json(paths(root,n,begin)[3])["roles"]
        for offset,(_,raw,known) in enumerate(roles):
            physical[begin+offset,raw] = 1
            physical[begin+offset,known] = 0
    orders = balanced_orders(physical,p["mode"],rng_for(root.name,n,"publication"))
    tmp = path.with_suffix(".tmp")
    with open(tmp,"wb") as f:
        np.save(f,orders)
    os.replace(tmp,path)


def finalize(root, split, smoke, deep=False):
    expected = quotas(split,smoke)
    counts, focuses, witnessed, appearance = Counter(),Counter(),Counter(),Counter()
    all_events, seen_events = Counter(),Counter()
    quality,layouts,acquisitions,repeated,severity = Counter(),Counter(),Counter(),Counter(),Counter()
    rejects = 0
    exp = {n: np.zeros((2,n),dtype=np.int64) for n in widths(smoke)}
    # Retain only 16 binary bytes per record for the cross-width ID audit.
    uid_table = np.empty(sum(expected.values()),dtype="V16")
    ancestor_count,episode_count,_ = dimensions(split,smoke)
    tmp = root/"manifest.partial"
    total = 0
    shard_hash = hashlib.sha256()
    with open(tmp,"wb") as out:
        for n in widths(smoke):
            p = read_plan(str(root/"build"/f"plan-{n}.npy"))
            for begin in range(0,len(p),BATCH):
                receipt = read_json(paths(root,n,begin)[-1])
                for kind in ("public","private"):
                    name = receipt[kind]
                    check = digest(root/name)
                    if check != receipt[kind+"_sha256"]:
                        raise IOError(f"archive hash mismatch {name}")
                    shard_hash.update(encode([name,check]))
                if len(receipt["rows"]) != min(BATCH,len(p)-begin):
                    raise ValueError("short shard")
                for j,row in enumerate(receipt["rows"],begin):
                    require_frozen_seasons(row,row["rejections"])
                    ep = Episode(split,n,*decode_record(*p[j]),smoke=smoke)
                    family = FAMILIES[j//(ancestor_count*episode_count)]
                    candidate = ancestors(split,n,family,ancestor_count)[j//episode_count%ancestor_count]
                    if (ep.family,ep.ancestor,ep.episode) != (family,candidate,j%episode_count):
                        raise ValueError("ancestor/episode multiplicity or split membership failed")
                    if row["id"] != ep.id or (row["K"],row["mode"],row["ancestor"],row["episode"],row["family"]) != (
                            ep.k,ep.mode,ep.ancestor,ep.episode,ep.family):
                        raise ValueError("receipt row disagrees with ancestor quota")
                    cell = (n,ep.k,ep.mode)
                    counts[cell] += 1
                    if row["focal"]:
                        layer,code,band = row["focal"]
                        focuses[cell+(layer,code)] += 1
                        appearance[cell+(layer,code,band)] += 1
                        witnessed[cell+(layer,code)] += row["focal_witness"]
                    raw,known = row["raw"],row["known"]
                    exp[n][0,raw] += 1
                    exp[n][1,known] += 1
                    for layer,kind,seen,band in row["events"]:
                        all_events[cell+(layer,kind)] += 1
                        seen_events[cell+(layer,kind)] += int(seen)
                        severity[(layer,kind,band or "structural")] += 1
                    for key,value in row["quality"].items():
                        quality[cell+(key,)] += int(value)
                    layouts[cell+(row["layout"],)] += 1
                    acquisitions[cell+(row["acquisition"],)] += 1
                    repeated[cell+(len(row["events"])-ep.k,)] += 1
                    rejects += row["rejections"]
                    private_keys = {"raw","known","events","focal_witness","layout","acquisition","quality","rejections","season_retry"}
                    out.write(encode({k:v for k,v in row.items() if k not in private_keys})+b"\n")
                    uid_table[total] = bytes.fromhex(row["id"])
                    total += 1
            if np.ptp(exp[n][0]) > 1 or np.ptp(exp[n][1]) > 1:
                raise ArithmeticError("published exposure failed")
        out.flush()
        os.fsync(out.fileno())
    if counts != expected:
        raise ValueError(f"cell mismatch actual={counts}, expected={expected}")
    uid_table.sort()
    if np.any(uid_table[1:] == uid_table[:-1]):
        raise ValueError("duplicate opaque identifier")
    reps = dimensions(split,smoke)[2]
    risk = []
    for (n,k,mode),cnt in expected.items():
        if not k:
            continue
        for layer,code in templates(mode):
            key = n,k,mode,layer,code
            if focuses[key] != reps:
                raise ValueError(f"focal coverage failed {key}: {focuses[key]}/{reps}")
            if reps == 96 and (code.startswith("L") or code == "S04"):
                if any(appearance[key+(s,)] != 32 for s in range(3)):
                    raise ValueError(f"imbalanced focal severity {key}")
            if reps == 96 and witnessed[key] < 12:
                risk.append([*key,witnessed[key]])
    def records(counter):
        return [[*key,value] for key,value in sorted(counter.items())]
    report = dict(profile=PROFILE,smoke=smoke,split=split,windows=total,cells=records(counts),
                  root_exposure={n:a.tolist() for n,a in exp.items()},focal_counts=records(focuses),
                  focal_witnessed=records(witnessed),low_focal_visibility=risk,
                  event_counts=records(all_events),events_with_direct_visible_witness=records(seen_events),
                  quality_flags=records(quality),layout_counts=records(layouts),acquisition_counts=records(acquisitions),
                  repeated_source_counts=records(repeated),event_severity_counts=records(severity),numerical_rejections=rejects,
                  archive_list_sha256=shard_hash.hexdigest(),calibration_sha256=digest(root.parent/"calibration.json"))
    if rejects:
        report["numerical_retry_seasons"] = "initial-background"
    os.replace(tmp,root/"manifest.jsonl")
    report["manifest_sha256"] = digest(root/"manifest.jsonl")
    if deep:
        deep_check(root)
    write_json(root/"audit.json",report)
    write_json(root/"_SUCCESS",dict(profile=PROFILE,windows=total,manifest_sha256=report["manifest_sha256"],
                                   audit_sha256=digest(root/"audit.json"),smoke=smoke))
    return report


def deep_check(root):
    current = None
    handle = None
    try:
        with (root/"manifest.jsonl").open(encoding="utf-8") as manifest:
            for line in manifest:
                row = json.loads(line)
                if current != row["shard"]:
                    if handle:
                        handle.close()
                    current = row["shard"]
                    handle = (root/current).open("rb")
                for name,keys in (("input",INPUTS),("target",TARGETS)):
                    handle.seek(row[name+"_offset"])
                    data = unpack(handle.read(row[name+"_size"]))
                    if set(data) != set(keys):
                        raise ValueError(f"damaged indexed {name} member")
    finally:
        if handle:
            handle.close()


def configure_worker():
    os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
    if os.environ.get("SSTER_GP_DEVICE","cpu").startswith("cuda"):
        import torch
        devices = torch.cuda.device_count()
        if not devices:
            raise RuntimeError("CUDA GP requested but no GPU is visible")
        rank = multiprocessing.current_process()._identity[-1]-1
        torch.cuda.set_device(rank % devices)


def generate(out, split, workers, smoke=False):
    out = Path(out)
    out.mkdir(exist_ok=True,parents=True)
    root = out/split
    for sub in ("build","receipts","shards","private"):
        (root/sub).mkdir(parents=True,exist_ok=True)
    with lock(root/"build.lock"):
        success = root/"_SUCCESS"
        if success.exists():
            doc = read_json(success)
            if doc["smoke"] != smoke or doc["manifest_sha256"] != digest(root/"manifest.jsonl") or doc["audit_sha256"] != digest(root/"audit.json"):
                raise IOError("existing release does not match requested profile or checksums")
            previous = read_json(root/"audit.json")
            if previous["numerical_rejections"] and previous.get("numerical_retry_seasons") != "initial-background":
                raise ValueError("existing release contains legacy numerical retries; audit and rebuild affected shards")
            print(f"{split} already published {doc['windows']} windows",flush=True)
            return
        os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
        with multiprocessing.get_context("spawn").Pool(workers,initializer=configure_worker) as pool:
            print(f"{split}: fitting/checking independent scale bank",flush=True)
            bank = calibration(out,smoke,pool)
            for n in widths(smoke):
                _,p_path,*_ = paths(root,n,0)
                print(f"{split} N={n}: matching frozen backgrounds and template quotas",flush=True)
                p = plan(p_path,split,n,smoke)
                starts = range(0,len(p),BATCH)
                jobs = ((root,split,smoke,n,b,min(b+BATCH,len(p)),bank["scales"]) for b in starts)
                for i,_ in enumerate(pool.imap_unordered(stage,jobs,chunksize=1),1):
                    if i == 1 or i%128 == 0 or i == len(starts):
                        print(f"{split} N={n}: physical chunks {i}/{len(starts)}",flush=True)
                print(f"{split} N={n}: assigning export channels",flush=True)
                prepare_orders(root,n,p,starts)
                for _ in pool.imap_unordered(publish_batch,((root,n,b) for b in starts),chunksize=1):
                    pass
            report = finalize(root,split,smoke)
            print(f"{split}: verified and published {report['windows']} windows",flush=True)
