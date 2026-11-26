"""Regenerate an example using its versioned, private event and stream record."""
import json
from functools import lru_cache
from pathlib import Path
import tarfile

import numpy as np

from .episodes import realize,require_frozen_seasons
from .ancestry import Roster, check_plan, resolve, sidecar, validate_world
from .planning import Episode, dimensions, encode
from . import round1
from .scm import FAMILIES, reserved_ancestors
from .storage import atomic, digest, pack, read_json, unpack


@lru_cache(maxsize=16)
def replay_roster(path,split,n,smoke,plan_sha256,ledger_sha256,changed,selection_sha256=None):
    """Cache only admitted IDs; never retain an open plan handle on Windows."""
    count = dimensions(split,smoke)[0]
    if changed:
        roster = resolve(split,n,FAMILIES,count)
    else:
        roster = {f:Roster(reserved_ancestors(split,n,f,count),(),()) for f in FAMILIES}
    if selection_sha256:
        round1.verify(path,n,roster)
    else:
        check_plan(path,split,n,smoke,roster)
    return {f:r.ids for f,r in roster.items()}


def verify_ancestor(root,ep):
    count, episodes, _ = dimensions(ep.split,ep.smoke)
    path = root/"build"/f"plan-{ep.n}.npy"
    binding = sidecar(path)
    marker = root/"_SUCCESS"
    expected = None
    selected = round1.is_first_round(root)
    selection_sha = digest(path.parent/f"round1-{ep.n}.json") if selected else None
    ledger_hash = digest(binding) if binding.exists() else None
    if marker.exists():
        if digest(root/"audit.json") != read_json(marker)["audit_sha256"]:
            raise IOError("replay audit disagrees with release marker")
        audit = read_json(root/"audit.json")
        expected = audit.get("ancestor_replacement_sha256",{}).get(str(ep.n))
        if (audit.get("selection_sha256",{}).get(str(ep.n)) != selection_sha or
                audit.get("training_edition") != (round1.EDITION if selected else None)):
            raise ValueError("replay selection differs from published training edition")
        if expected and ledger_hash != expected:
            raise ValueError("published ancestor replacement ledger missing or altered")
    if binding.exists() or marker.exists():
        changed = binding.exists() and any(r["replaced"] or r["rejected"] for r in read_json(binding)["families"].values())
        if marker.exists() and bool(expected) != changed:
            raise ValueError("replay ancestor map differs from published binding")
        roster = replay_roster(path,ep.split,ep.n,ep.smoke,digest(path),ledger_hash,changed,selection_sha)
        ids = roster[ep.family]
        if ep.ancestor not in ids or not 0 <= ep.episode < episodes:
            raise ValueError("replay record belongs to no frozen ancestor or episode slot")
        world_index = FAMILIES.index(ep.family)*count+ids.index(ep.ancestor)
        p = np.load(path,mmap_mode="r",allow_pickle=False)
        if selected:
            start = world_index*round1.PER_ANCESTOR
            choices = p["episode"][start:start+round1.PER_ANCESTOR]
            offset = int(np.searchsorted(choices,ep.episode))
            if offset == len(choices) or choices[offset] != ep.episode:
                raise ValueError("episode was not selected for first-round training")
            index = start + offset
        else:
            index = world_index*episodes+ep.episode
        if Episode(ep.split,ep.n,*encode(*p[index]),smoke=ep.smoke) != ep:
            raise ValueError("replay ticket disagrees with frozen plan")
    elif ep.ancestor not in reserved_ancestors(ep.split,ep.n,ep.family,count):
        raise ValueError("unbound replay ancestor outside the reserved roster")
    validate_world(ep.split,ep.n,ep.family,ep.ancestor)


def replay(root,shard,uid,save=None):
    root = Path(root)
    tag = Path(shard).name
    with tarfile.open(root/"private"/tag,"r:") as tar:
        meta = json.load(tar.extractfile(uid+".replay.json"))
    require_frozen_seasons(meta,meta["retry"])
    ep = Episode(meta["split"],meta["N"],meta["family"],meta["ancestor"],meta["episode"],meta["K"],
                 meta["mode"],tuple(meta["focal"]) if meta["focal"] else None,meta["smoke"])
    if ep.id != uid:
        raise ValueError("replay identifier mismatch")
    verify_ancestor(root,ep)
    inputs,targets,_,traces = realize(ep,meta["scale"],meta["retry"],audit=True)
    order = np.argsort(meta["physical_to_exported"])
    with tarfile.open(root/"shards"/tag,"r:") as tar:
        for suffix,mapping in (("inputs",inputs),("targets",targets)):
            public = unpack(tar.extractfile(f"{uid}.{suffix}.npz").read())
            for key,a in mapping.items():
                actual = a if key in ("time","Y","Y_known") else np.take(a,order,axis=-1)
                if not np.array_equal(actual,public[key]):
                    raise ArithmeticError(f"regenerated {key} differs from the published array")
    if save:
        atomic(save,pack(traces))
    return uid
