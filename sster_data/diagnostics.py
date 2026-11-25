"""Normal-prior diagnostics and paired dense-vs-interpolated GP comparisons."""
from collections import defaultdict
import json
from pathlib import Path
import tarfile

import numpy as np

from . import catalog
from .episodes import require_frozen_seasons
from .planning import Episode
from .scm import FAMILIES, rng_for, solve, world
from .storage import read_json, unpack, write_json


METRICS = ("acf1","acf4","acf16","power_0_0625","power_0625_25","power_25_5")
KERNELS = ("rbf","periodic","rq","matern32","constant","white","linear","white_plus_rbf")
TRENDS = ("down","up","constant","multiple","arima")
SEASONS = ("none","sine","square","triangle","wavelet")
LEVELS = ("level_0","level_1","level_2","level_3")


def measures(values):
    x = np.asarray(values,dtype=float)
    if x.ndim == 1:
        x = x[:,None]
    x = x - x.mean(0)
    energy = (x*x).sum(0)
    frequencies = np.fft.rfftfreq(len(x))[:,None]
    psd = np.abs(np.fft.rfft(x,axis=0))**2
    psd[0] = 0
    # One-sided periodogram retaining the DC and Nyquist conventions.
    psd[1:-1 if len(x)%2 == 0 else None] *= 2
    den = psd.sum(0)
    valid = energy > 1e-16
    values = []
    for lag in (1,4,16):
        values.append(np.divide((x[:-lag]*x[lag:]).sum(0),energy,out=np.zeros_like(energy),where=valid))
    for b,e in ((0,.0625),(.0625,.25),(.25,1)):
        values.append(np.divide(np.where((frequencies>=b)&(frequencies<e),psd,0).sum(0),den,
                                out=np.zeros_like(den),where=valid))
    return np.stack(values,axis=1)


def dense_comparison(seed,repetitions,length):
    if length <= 256:
        raise ValueError("dense reference must have more than 256 physical ticks")
    out = {}
    for name in KERNELS:
        pairs = []
        for j in range(repetitions):
            rr = rng_for("dense-gp",seed,name,j)
            family = FAMILIES[j%4]
            wld = world("diagnostic",1,family,j)
            warm = wld.warm
            # Joint sampling makes the dense and sparse branches observations
            # of the same continuous-time GP (including identical coincident knots).
            grid = np.arange(-warm,length,dtype=float)
            knots = np.linspace(-warm,length-1,256)
            ticks, inverse = np.unique(np.r_[grid,knots],return_inverse=True)
            distance = np.abs(ticks[:,None]-ticks)
            ell = catalog.logu(rr,2,length/2)
            if name in ("rbf","white_plus_rbf"):
                cov = np.exp(-.5*(distance/ell)**2)
                if name == "white_plus_rbf":
                    cov = (cov+np.eye(len(cov)))*.5
            elif name == "periodic":
                period = catalog.logu(rr,2,length/2)
                cov = np.exp(-2*(np.sin(np.pi*distance/period)/ell)**2)
            elif name == "rq":
                alpha = rr.uniform(.25,4)
                cov = (1+(distance/ell)**2/(2*alpha))**(-alpha)
            elif name == "matern32":
                d = np.sqrt(3)*distance/ell
                cov = (1+d)*np.exp(-d)
            elif name == "constant":
                cov = np.ones_like(distance)
            elif name == "white":
                cov = np.eye(len(ticks))
            else:
                u = np.clip(ticks/(length-1),0,1)
                cov = (1+np.outer(u,u))/2.
            draw = catalog.factor(cov) @ rr.normal(size=len(ticks))
            dense = draw[inverse[:len(grid)]]
            sparse = np.interp(grid,knots,draw[inverse[len(grid):]])
            bg = catalog.normal_background(rr,1,warm,length)
            mean = np.clip(bg.trends[:,0]+bg.seasonal[:,0],-2,2)
            innovation = bg.b[:,0]-np.clip(mean+bg.gp[:,0],-4,4)
            metric = []
            for gp in (dense,sparse):
                drive = (np.clip(mean+2*np.tanh(gp),-4,4)+innovation)[:,None].copy()
                state = solve(drive,np.empty((0,0)),False,*wld.args)[0][warm:,0]
                metric.append(np.vstack([measures(gp[warm:]),measures(state)]))
            pairs.append(metric)
        values = np.array(pairs)  # draws x branch x signal x metric
        summary = {}
        for si,signal in enumerate(("raw_gp","final_healthy_state")):
            a,b = values[:,:,si,:].mean(0)
            summary[signal] = dict(dense=a.tolist(),sparse=b.tolist(),signed_difference=(b-a).tolist(),
                                   flag_over_point_one=[m for m,d in zip(METRICS,b-a) if abs(d)>.10])
        out[name] = summary
    return out


def envelope(blocks,seed,replicates=1000):
    if not blocks:
        return None
    counts = np.array([a for a,b in blocks],dtype=float)
    totals = np.array([b for a,b in blocks],dtype=float)
    rng = rng_for("diagnostic-bootstrap",seed)
    summaries = []
    for _ in range(replicates):
        pick = rng.integers(0,len(blocks),len(blocks))
        total = totals[pick].sum()
        if total:
            summaries.append(counts[pick].sum(axis=0)/total)
    if not summaries:
        return None
    return dict(clusters=len(blocks),denominator=int(totals.sum()),mean=(counts.sum(0)/totals.sum()).tolist(),
                ci95=np.quantile(summaries,[.025,.975],axis=0).T.tolist())


def metadata_kinds(ep,retry):
    _,_,_,tr,nr = catalog.streams(ep.rng("background",retry),5)
    wld = world(ep.split,ep.n,ep.family,ep.ancestor)
    _,length,_ = ep.clock(wld.lag)
    season = [component["kind"] for component in
              catalog.normal_seasons(ep.rng("background"),ep.n,wld.warm,length).components]
    trend = [r.choice(TRENDS,p=[.2,.2,.2,.3,.1]) for r in catalog.streams(tr,ep.n)]
    noise = []
    for r in catalog.streams(nr,ep.n):
        r.integers(4)  # innovation law precedes the discretized scale
        noise.append(LEVELS[int(r.integers(4))])
    return np.array(list(zip(trend,season,noise)))


def released_backgrounds(root):
    # Every 8192nd physical row is retained as a replay audit, independent of
    # event mode, visibility, family dynamics and GP spectrum.
    groups = defaultdict(lambda:defaultdict(list))
    labels = TRENDS+SEASONS+LEVELS
    count = 0
    for path in sorted((root/"private").glob("n*-*.tar")):
        if int(path.stem.split("-")[-1])%8192:
            continue
        with tarfile.open(path,"r:") as archive:
            for member in archive.getnames():
                if not member.endswith(".trace.npz"):
                    continue
                uid = member.split(".")[0]
                meta = json.load(archive.extractfile(uid+".replay.json"))
                require_frozen_seasons(meta,meta["retry"])
                ep = Episode(meta["split"],meta["N"],meta["family"],meta["ancestor"],meta["episode"],meta["K"],
                             meta["mode"],meta["focal"],meta["smoke"])
                key = (ep.n,ep.family,ep.ancestor)
                kinds = metadata_kinds(ep,meta["retry"])
                target = sorted({e["target"] for e in meta["events"] if e["kind"][0]=="S"})
                for tier,indices in (("all_nodes",np.arange(ep.n)),("seasonal_fault_roots",target)):
                    subset = kinds[indices] if len(indices) else kinds[:0]
                    counts = np.array([(subset[:,axis]==name).sum() for axis,names in enumerate((TRENDS,SEASONS,LEVELS))
                                       for name in names])
                    groups[tier][key].append((counts,len(subset)))
                arr = unpack(archive.extractfile(member).read())["normal"]
                wld = world(ep.split,ep.n,ep.family,ep.ancestor)
                _,length,_ = ep.clock(wld.lag)
                x = measures(arr[:length])
                groups["paired_healthy_state"][key].append((x.sum(0),len(x)))
                # Innovation diagnostics on the same actual physical clocks.
                nr = catalog.streams(ep.rng("background",meta["retry"]),5)[4]
                ticks = np.arange(-wld.warm,length)
                noise = np.stack([catalog.innovations(r,ticks,length)[wld.warm:] for r in catalog.streams(nr,ep.n)],axis=1)
                noise_metrics = measures(noise)
                groups["normal_innovations"][key].append((noise_metrics.sum(0),len(noise_metrics)))
                count += 1
    result = {"audit_episodes":count,"prior_columns":list(labels),"metric_columns":list(METRICS)}
    for name,ancestors in groups.items():
        blocks = [(sum(r[0] for r in rows),sum(r[1] for r in rows)) for rows in ancestors.values()]
        result[name] = envelope(blocks,name)
    return result


def diagnose(out,split,repetitions=64,length=512):
    root = Path(out)/split
    release = read_json(root/"_SUCCESS")
    result = dict(split=split,manifest_sha256=release["manifest_sha256"],repetitions_per_kernel=repetitions,
                  dense_length=length,metrics=list(METRICS),paired_gp=dense_comparison(17,repetitions,length),
                  released_backgrounds=released_backgrounds(root))
    path = root/"background_diagnostics.json"
    write_json(path,result)
    return path
