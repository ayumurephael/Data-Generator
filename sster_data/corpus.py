"""Resumable, deterministic width-at-a-time corpus builder."""
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import tarfile

import numpy as np

from .ancestry import check_plan, resolve, sidecar as ancestry_sidecar
from . import round1
from .balance import balanced_orders
from .calibration import calibration
from .episodes import NumericalFailure, realize, require_frozen_seasons
from .planning import dimensions, encode as decode_record, Episode, MODES, plan, quotas, templates, widths
from .scm import FAMILIES, PROFILE, rng_for
from .storage import Archive, atomic, digest, encode, lock, pack, read_json, unpack, write_json

BATCH = 64
INPUTS = ("values","observed","time","channel_valid")
TARGETS = ("Y","Y_known","R","R_known","E")
SHARD_LIMIT = 256*1024**2
FINALIZE_READERS = 4


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


def write_native_episode(archive, ep, scale, audit):
    failures = []
    for retry in range(50):
        try:
            inp,tgt,meta,traces = realize(ep,scale,retry,audit=audit)
            break
        except (NumericalFailure, np.linalg.LinAlgError) as err:
            failures.append(str(err))
    else:
        raise RuntimeError(f"numerical rejection exhausted for {ep.id}: {failures}")
    meta.update(rejections=failures,profile=PROFILE,smoke=ep.smoke)
    archive.add(ep.id+".npz",pack({**inp, **tgt}))
    archive.add(ep.id+".json",encode(meta))
    if traces is not None:
        archive.add(ep.id+".trace.npz",pack(traces))
    known = np.flatnonzero(tgt["R"] & tgt["R_known"]).tolist()
    roots = np.flatnonzero(tgt["R"]).tolist()
    return [ep.id,roots,known]


def stage(job):
    root,split,smoke,n,begin,end,scales = job
    tag,plan_path,stage_path,roles_path,receipt = paths(root,n,begin)
    p = read_plan(str(plan_path))
    if receipt.exists():
        if not roles_path.exists():
            raise IOError("committed shard missing its exposure receipt")
        rows = read_json(receipt)["rows"]
        if len(rows) != end-begin:
            raise ValueError(f"short committed shard {tag}")
        for serial,row in enumerate(rows,begin):
            ep = Episode(split,n,*decode_record(*p[serial]),smoke=smoke)
            if (row["id"],row["K"],row["mode"],row["focal"]) != (ep.id,ep.k,ep.mode,list(ep.focus) if ep.focus else None):
                raise ValueError(f"committed shard disagrees with frozen plan {tag}")
            require_frozen_seasons(row,row["rejections"])
        return
    if roles_path.exists() and stage_path.exists():
        roles = read_json(roles_path)
        if digest(stage_path) != roles["sha256"]:
            raise IOError(f"damaged physical staging archive {tag}")
        tickets = [Episode(split,n,*decode_record(*p[j]),smoke=smoke) for j in range(begin,end)]
        if [r[0] for r in roles["roles"]] != [ep.id for ep in tickets]:
            raise ValueError(f"staged ancestor/episode IDs disagree with frozen plan {tag}")
        with tarfile.open(stage_path,"r:") as archive:
            for ep in tickets:
                meta = json.loads(read_member(archive,ep.id+".json"))
                if (meta["K"],meta["mode"],meta["focal"]) != (ep.k,ep.mode,list(ep.focus) if ep.focus else None):
                    raise ValueError(f"staged ticket disagrees with frozen plan {tag}")
        return
    roles = []
    with Archive(stage_path) as archive:
        for serial in range(begin,end):
            ep = Episode(split,n,*decode_record(*p[serial]),smoke=smoke)
            roles.append(write_native_episode(archive, ep, scales[ep.family], serial%8192==0))
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


def _verified_receipt(root, n, begin):
    receipt = read_json(paths(root,n,begin)[-1])
    for kind in ("public","private"):
        name = receipt[kind]
        if digest(root/name) != receipt[kind+"_sha256"]:
            raise IOError(f"archive hash mismatch {name}")
    return begin,receipt


def _ordered_receipts(root, n, starts, readers):
    # Bound decoded receipts and open archives independently of the split size.
    # Only the consumer folds verified hashes, rows, and exceptions into plan order.
    pending = deque()
    for begin in starts:
        pending.append(readers.submit(_verified_receipt,root,n,begin))
        if len(pending) == FINALIZE_READERS:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def finalize(root, split, smoke, deep=False, roster_by_width=None, pool=None):
    first_round = round1.is_first_round(root)
    if first_round and (smoke or split != "train"):
        raise ValueError("first-round quotas apply only to production training")
    existing = None
    if (root/"_SUCCESS").exists():
        marker = read_json(root/"_SUCCESS")
        if marker.get("training_edition") != (round1.EDITION if first_round else None):
            raise ValueError("published marker disagrees with the training edition")
        if marker["audit_sha256"] != digest(root/"audit.json") or marker["manifest_sha256"] != digest(root/"manifest.jsonl"):
            raise IOError("published marker does not match audit or manifest")
        existing = read_json(root/"audit.json")
    expected = round1.quotas() if first_round else quotas(split,smoke)
    counts, focuses, witnessed, appearance = Counter(),Counter(),Counter(),Counter()
    all_events, seen_events = Counter(),Counter()
    quality,layouts,acquisitions,repeated,severity = Counter(),Counter(),Counter(),Counter(),Counter()
    rejects = 0
    exp = {n: np.zeros((2,n),dtype=np.int64) for n in widths(smoke)}
    # Retain only 16 binary bytes per record for the cross-width ID audit.
    uid_table = np.empty(sum(expected.values()),dtype="V16")
    ancestor_count,episode_count,_ = dimensions(split,smoke)
    if first_round:
        episode_count = round1.PER_ANCESTOR
    tmp = root/"manifest.partial"
    total = 0
    shard_hash = hashlib.sha256()
    repairs = {}
    with open(tmp,"wb") as out, ThreadPoolExecutor(max_workers=FINALIZE_READERS) as readers:
        for n in widths(smoke):
            path = root/"build"/f"plan-{n}.npy"
            roster = resolve(split,n,FAMILIES,ancestor_count,pool) if roster_by_width is None else roster_by_width[n]
            p = round1.verify(path,n,roster) if first_round else check_plan(path,split,n,smoke,roster)
            if any(r.replaced for r in roster.values()):
                repairs[str(n)] = digest(ancestry_sidecar(path))
            starts = range(0,len(p),BATCH)
            for begin,receipt in _ordered_receipts(root,n,starts,readers):
                for kind in ("public","private"):
                    shard_hash.update(encode([receipt[kind],receipt[kind+"_sha256"]]))
                if len(receipt["rows"]) != min(BATCH,len(p)-begin):
                    raise ValueError("short shard")
                for j,row in enumerate(receipt["rows"],begin):
                    require_frozen_seasons(row,row["rejections"])
                    ep = Episode(split,n,*decode_record(*p[j]),smoke=smoke)
                    family = FAMILIES[j//(ancestor_count*episode_count)]
                    candidate = roster[family].ids[j//episode_count%ancestor_count]
                    if ((ep.family,ep.ancestor) != (family,candidate)
                            or (not first_round and ep.episode != j%episode_count)):
                        raise ValueError("ancestor/episode multiplicity or split membership failed")
                    if row["id"] != ep.id or (row["K"],row["mode"],row["ancestor"],row["episode"],row["family"],row["focal"]) != (
                            ep.k,ep.mode,ep.ancestor,ep.episode,ep.family,list(ep.focus) if ep.focus else None):
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
    reps = round1.FOCAL if first_round else dimensions(split,smoke)[2]
    risk = []
    for (n,k,mode),cnt in expected.items():
        if not k:
            continue
        for layer,code in templates(mode):
            key = n,k,mode,layer,code
            if focuses[key] != reps:
                raise ValueError(f"focal coverage failed {key}: {focuses[key]}/{reps}")
            if reps in (48,96) and (code.startswith("L") or code == "S04"):
                if any(appearance[key+(s,)] != reps//3 for s in range(3)):
                    raise ValueError(f"imbalanced focal severity {key}")
            if reps in (48,96) and witnessed[key] < 12:
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
    if repairs:
        report["ancestor_replacement_sha256"] = repairs
    edition_fields = {}
    if first_round:
        if any(np.ptp(a[0]) for a in exp.values()):
            raise ArithmeticError("first-round raw exposure is not exactly equal across slots")
        edition_fields["training_edition"] = round1.EDITION
        report.update(edition_fields)
        report["selection_sha256"] = {n:digest(root/"build"/f"round1-{n}.json") for n in widths(False)}
    if existing is not None and repairs != existing.get("ancestor_replacement_sha256",{}):
        raise ValueError("published ancestor replacement binding differs from verified plans")
    report["manifest_sha256"] = digest(tmp)
    if existing is not None and hashlib.sha256(encode(report)+b"\n").hexdigest() != marker["audit_sha256"]:
        raise ValueError("reconstructed audit or manifest differs from the published release")
    os.replace(tmp,root/"manifest.jsonl")
    if deep:
        deep_check(root)
    write_json(root/"audit.json",report)
    write_json(root/"_SUCCESS",dict(profile=PROFILE,windows=total,manifest_sha256=report["manifest_sha256"],
                                   audit_sha256=digest(root/"audit.json"),smoke=smoke,**edition_fields))
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


class WorkerDeviceError(ValueError):
    """An explicit GP worker device map cannot run on this host."""


def validate_worker_devices(devices):
    if not devices or any(type(index) is not int or index < 0 for index in devices):
        raise WorkerDeviceError("--worker-devices requires one or more nonnegative CUDA indices")
    try:
        import torch
    except ImportError as exc:
        raise WorkerDeviceError("--worker-devices requires PyTorch with CUDA support") from exc
    if not torch.cuda.is_available():
        raise WorkerDeviceError("--worker-devices requires CUDA; no CUDA device is available")
    count = torch.cuda.device_count()
    if max(devices) >= count:
        raise WorkerDeviceError(f"--worker-devices index {max(devices)} is outside the {count} visible CUDA devices")


def configure_worker(worker_devices=None, next_slot=None):
    os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
    if worker_devices is not None:
        import torch
        with next_slot.get_lock():
            index = worker_devices[next_slot.value % len(worker_devices)]
            next_slot.value += 1
        torch.cuda.set_device(index)
        os.environ["SSTER_GP_DEVICE"] = f"cuda:{index}"
    elif os.environ.get("SSTER_GP_DEVICE","cpu").startswith("cuda"):
        import torch
        devices = torch.cuda.device_count()
        if not devices:
            raise RuntimeError("CUDA GP requested but no GPU is visible")
        rank = multiprocessing.current_process()._identity[-1]-1
        torch.cuda.set_device(rank % devices)


def generate(out, split, workers, smoke=False, worker_devices=None, training_edition=None,
             parent_source=None, extra_builds=()):
    if training_edition not in (None, round1.EDITION) or (training_edition and (split != "train" or smoke)):
        raise ValueError("unknown edition or incompatible split")
    if parent_source is not None and not training_edition:
        raise ValueError("parent archives are only used for first-round training")
    if worker_devices is not None:
        worker_devices = tuple(worker_devices)
        validate_worker_devices(worker_devices)
    out = Path(out)
    out.mkdir(exist_ok=True,parents=True)
    root = out/split
    if parent_source is not None and Path(parent_source).resolve() == out.resolve():
        raise ValueError("first-round release requires an independent output directory")
    for sub in ("build","receipts","shards","private"):
        (root/sub).mkdir(parents=True,exist_ok=True)
    with lock(root/"build.lock"):
        round1.bind(root, training_edition)
        ctx = multiprocessing.get_context("spawn")
        ancestor_count = dimensions(split,smoke)[0]
        success = root/"_SUCCESS"
        if success.exists():
            doc = read_json(success)
            if doc.get("training_edition") != training_edition:
                raise ValueError("existing release uses a different training edition")
            if doc["smoke"] != smoke or doc["manifest_sha256"] != digest(root/"manifest.jsonl") or doc["audit_sha256"] != digest(root/"audit.json"):
                raise IOError("existing release does not match requested profile or checksums")
            previous = read_json(root/"audit.json")
            if previous.get("training_edition") != training_edition:
                raise ValueError("published audit disagrees with the training edition")
            if previous["numerical_rejections"] and previous.get("numerical_retry_seasons") != "initial-background":
                raise ValueError("existing release contains legacy numerical retries; audit and rebuild affected shards")
            repairs = {}
            with ctx.Pool(workers) as validation_pool:
                for n in widths(smoke):
                    path = root/"build"/f"plan-{n}.npy"
                    roster = resolve(split,n,FAMILIES,ancestor_count,validation_pool)
                    if training_edition:
                        round1.verify(path,n,roster)
                        if previous.get("selection_sha256",{}).get(str(n)) != digest(path.parent/f"round1-{n}.json"):
                            raise ValueError("published ticket selection does not match the verified plans")
                    else:
                        check_plan(path,split,n,smoke,roster)
                    if any(r.replaced for r in roster.values()):
                        repairs[str(n)] = digest(ancestry_sidecar(path))
            if repairs != previous.get("ancestor_replacement_sha256",{}):
                raise ValueError("published ancestor replacement binding differs from verified plans")
            print(f"{split} already published {doc['windows']} windows",flush=True)
            return
        os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
        # A fresh counter per pool makes sequential and separately launched splits start at slot zero.
        initargs = () if worker_devices is None else (worker_devices,ctx.Value("q",0))
        with ctx.Pool(workers,initializer=configure_worker,initargs=initargs) as pool:
            print(f"{split}: fitting/checking independent scale bank",flush=True)
            if parent_source is not None:
                round1.reuse_calibration(out, parent_source)
            bank = calibration(out,smoke,pool)
            stage_worker = stage
            if parent_source is not None:
                from . import reuse
                if digest(out/"calibration.json") != digest(Path(parent_source)/"calibration.json"):
                    raise ValueError("cannot reuse examples with different calibration")
                stage_worker = partial(reuse.stage,source_root=Path(parent_source)/"train",extra_builds=extra_builds)
            roster_by_width = {}
            for n in widths(smoke):
                _,p_path,*_ = paths(root,n,0)
                print(f"{split} N={n}: matching frozen backgrounds and template quotas",flush=True)
                roster = resolve(split,n,FAMILIES,ancestor_count,pool)
                roster_by_width[n] = roster
                p = (round1.prepare(p_path,n,roster,parent_source) if training_edition else
                     plan(p_path,split,n,smoke,roster))
                starts = range(0,len(p),BATCH)
                jobs = ((root,split,smoke,n,b,min(b+BATCH,len(p)),bank["scales"]) for b in starts)
                for i,_ in enumerate(pool.imap_unordered(stage_worker,jobs,chunksize=1),1):
                    if i == 1 or i%128 == 0 or i == len(starts):
                        print(f"{split} N={n}: physical chunks {i}/{len(starts)}",flush=True)
                print(f"{split} N={n}: assigning export channels",flush=True)
                prepare_orders(root,n,p,starts)
                for _ in pool.imap_unordered(publish_batch,((root,n,b) for b in starts),chunksize=1):
                    pass
            report = finalize(root,split,smoke,roster_by_width=roster_by_width)
            print(f"{split}: verified and published {report['windows']} windows",flush=True)
