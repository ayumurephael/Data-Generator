"""Regenerate an example using its versioned, private event and stream record."""
import json
from pathlib import Path
import tarfile

import numpy as np

from .episodes import realize
from .planning import Episode
from .storage import atomic, pack, unpack


def replay(root,shard,uid,save=None):
    root = Path(root)
    tag = Path(shard).name
    with tarfile.open(root/"private"/tag,"r:") as tar:
        meta = json.load(tar.extractfile(uid+".replay.json"))
    ep = Episode(meta["split"],meta["N"],meta["family"],meta["ancestor"],meta["episode"],meta["K"],
                 meta["mode"],tuple(meta["focal"]) if meta["focal"] else None,meta["smoke"])
    if ep.id != uid:
        raise ValueError("replay identifier mismatch")
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
