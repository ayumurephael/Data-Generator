from pathlib import Path
import platform

import numpy as np

from .catalog import normal_background
from .episodes import healthy_path
from .planning import widths
from .scm import FAMILIES, PROFILE, stream, world
from .storage import digest, lock, read_json, write_json


def calibrate_job(args):
    n,family,ancestor,length = args
    wld = world("normal-bank",n,family,ancestor)
    bg = normal_background(stream("normal-bank",n,family,ancestor,0,"episode/background"),n,wld.warm,length)
    x = healthy_path(wld,bg)[wld.warm:]
    if not np.isfinite(x).all() or np.max(abs(x)) > 256 or x.std(0).max() < .001:
        raise ArithmeticError("invalid independent normal-bank ancestor")
    return family,x.ravel()


def fit(x):
    q1,q3 = np.quantile(x,[.25,.75],method="linear")
    return float(max(.001,(q3-q1)/1.349))


def calibration(out, smoke, pool):
    path = Path(out)/"calibration.json"
    with lock(path.with_suffix(".lock")):
        if not path.exists():
            jobs = []
            for fi,family in enumerate(FAMILIES):
                for ni,n in enumerate(widths(smoke)):
                    count = 1 if smoke else 9+int(ni == fi)
                    jobs.extend((n,family,i,256 if smoke else 2048) for i in range(count))
            data = {f: [] for f in FAMILIES}
            for family,x in pool.imap(calibrate_job,jobs,chunksize=1):
                data[family].append(x)
            combined = {f:np.concatenate(v) for f,v in data.items()}
            write_json(path,dict(profile=PROFILE,smoke=smoke,mode="normal-bank",jobs=jobs,
                                 scales={f:fit(v) for f,v in combined.items()},pooled=fit(np.concatenate(list(combined.values()))),
                                 numpy=np.__version__))
            path.with_suffix(".sha256").write_text(digest(path),encoding="ascii")
        doc = read_json(path)
        if doc["profile"] != PROFILE or doc["smoke"] != smoke:
            raise ValueError("existing corpus profile differs from requested run")
        stamp = path.with_suffix(".sha256")
        if not stamp.exists() or stamp.read_text(encoding="ascii").strip() != digest(path):
            raise IOError("calibration digest missing or invalid")
        return doc
