"""TimeRCD waveform catalogue and the versioned SSTER direct-state driving process."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import multiprocessing as mp
import os

import numpy as np
import pywt
from scipy.signal import lfilter


MOTHERS = ("db4", "sym5", "coif3", "haar", "bior1.3", "dmey")
MIN_LENGTH = {f"L{i:02}": d for i, d in enumerate(
    (3, 3, 8, 8, 4, 4, 1, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6), 1)}
MIN_LENGTH.update({f"S{i:02}": 8 for i in range(1, 21)})


def logu(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def signed(rng):
    return 1 if rng.integers(2) else -1


def rounded(value):
    return int(np.floor(value + 0.5))


def streams(rng, count):
    return [np.random.default_rng(int(v)) for v in rng.integers(0, 2**63, count, dtype=np.uint64)]


def triangle(n, duration):
    mid = (duration - 1) // 2
    width = max(1, min(mid, duration - 1 - mid))
    return np.maximum(1 - np.abs(n - mid) / width, 0)


def local_wave(kind, length, rng, amplitude):
    if kind not in MIN_LENGTH or kind[0] != "L" or length < MIN_LENGTH[kind]:
        raise ValueError(f"invalid local template {kind}/{length}")
    k = int(kind[1:])
    a = abs(float(amplitude))
    n = np.arange(length)
    mid = (length - 1) // 2
    if k <= 2:
        return (a if k == 1 else -a) * triangle(n, length)
    if k <= 4:
        count = int(rng.integers(2, min(4, length // 3) + 1))
        chunks = np.array_split(n, count)
        burst = np.concatenate([rng.uniform(.5, 1) * triangle(np.arange(len(c)), len(c)) for c in chunks])
        return (a if k == 3 else -a) * burst
    if k <= 6:
        flank = max(1, length // 3)
        return (a if k == 5 else -a) * np.minimum(1, np.minimum(n, length-1-n)/flank)
    if k == 7:
        if length != 1:
            raise ValueError("single-point shock must occupy one tick")
        return np.array([signed(rng)*a])
    if k <= 9:
        return (a if k == 8 else -a)/(1 + np.exp(-rng.uniform(4,12)*(n-mid)/length))
    if k <= 11:
        return (a if k == 10 else -a)*(1-np.cos(np.pi*n/length))/2
    if k <= 15:
        fast, slow = max(.5, length/12), max(1., length/3)
        rise, fall = (fast, slow) if k in (12,14) else (slow,fast)
        return (a if k <= 13 else -a)*np.where(n < mid, -np.expm1(-n/rise), np.exp(-(n-mid)/fall))
    if k <= 19:
        d = max(3, length//2)
        spike, after = ((1,-1),(-1,1),(1,1),(-1,-1))[k-16]
        return np.r_[spike*a*triangle(np.arange(d), d), np.full(length-d, after*a*rng.uniform(.5,1.5))]
    return a*np.sin(2*np.pi*rng.uniform(.18,.45)*n + rng.uniform(0,2*np.pi))


@lru_cache(maxsize=6)
def mother(name):
    table = pywt.Wavelet(name).wavefun(level=10)
    psi, grid = table[-2], table[-1]  # psi_r for biorthogonal families
    return 2*(grid-grid[0])/(grid[-1]-grid[0])-1, psi/np.max(np.abs(psi))


def new_atom(rng, duration, begin=0, end=None, width_max=None):
    return dict(name=str(rng.choice(MOTHERS)), center=float(rng.uniform(begin, end if end is not None else duration-1)),
                width=logu(rng,2, max(2, width_max if width_max is not None else max(3,duration/8))),
                amp=signed(rng)*rng.uniform(.2,1))


def season_params(rng, duration):
    kind = str(rng.choice(["none","sine","square","triangle","wavelet"],p=[.3,.3,.05,.05,.3]))
    if kind == "none":
        return dict(kind=kind, parts=[])
    if kind == "wavelet":
        return dict(kind=kind, parts=[new_atom(rng,duration) for _ in range(rng.integers(1,5))])
    harmonic = bool(rng.integers(2))
    count = int(rng.integers(1,4))
    multiples = [2.,3.,4.] if harmonic else np.sqrt([2.,3.,5.])
    orders = np.r_[1.,rng.choice(multiples, count-1, replace=False)]
    lo,hi = (4.,32.) if rng.integers(2) else (32.,max(32.,duration/2))
    # Preserve the category, component count and Bernoulli modulation choices.
    # The lower endpoint enforces both carrier and active upper sideband Nyquist limits.
    active = (rng.random(count)<.25) & (kind=="sine")
    ratios = np.exp(rng.uniform(np.log(2.),np.log(8.),count))
    lower = max(lo,np.nextafter(2*np.max(orders+active/ratios),np.inf))
    period = logu(rng,lower,hi)
    parts = []
    for order,am,r in zip(orders,active,ratios):
        parts.append(dict(f=float(order/period),order=float(order),amp=rng.uniform(.2,1)/(order if harmonic else 1),
                          phase=rng.uniform(0,2*np.pi),start=rng.random(),duty=rng.uniform(.2,.8),
                          depth=rng.uniform(.1,.7) if am else 0.,mod=1/(period*r),
                          mphase=rng.uniform(0,2*np.pi),wave=kind))
    return dict(kind=kind,parts=parts,harmonic=harmonic,base=1/period)


def render(spec, t):
    out = np.zeros(len(t),dtype=np.float64)
    for p in spec["parts"]:
        if spec["kind"]=="wavelet":
            grid,psi=mother(p["name"])
            out += p["amp"]*np.interp((t-p["center"])/p["width"],grid,psi,left=0,right=0)
        elif p["wave"]=="sine":
            out += p["amp"]*(1+p["depth"]*np.sin(2*np.pi*p["mod"]*t+p["mphase"]))*np.sin(2*np.pi*p["f"]*t+p["phase"])
        else:
            phase=np.mod(p["f"]*t+p["phase"]/(2*np.pi)-p["start"],1)
            inside=phase<p["duty"]
            out += p["amp"]*inside*(1 if p["wave"]=="square" else 1-np.abs(2*phase/p["duty"]-1))
    return out


def trend(rng,t,duration):
    kind=str(rng.choice(["down","up","constant","multiple","arima"],p=[.2,.2,.2,.3,.1]))
    x=np.clip(t,0,duration-1)
    intercept=rng.uniform(-1,1)
    det=np.full(t.shape,intercept)
    if kind in ("down","up"):
        u=x/(duration-1)
        det+=(1 if kind=="up" else -1)*rng.uniform(.1,1)*(2*u-1 if rng.integers(2) else np.expm1(u))
    elif kind=="multiple":
        det+=rng.uniform(-2,2)*x/(duration-1)
        for knot in np.sort(rng.choice(np.arange(1,duration-1),rng.integers(1,4),replace=False)):
            det+=rng.uniform(-2,2)*np.maximum(x-knot,0)/(duration-1)
    weight=1. if kind=="arima" else (rng.uniform(.05,.25) if rng.random()<.2 else 0.)
    if not weight:
        return det
    ar=rng.uniform(-1,1,rng.integers(3))
    ma=rng.uniform(-1,1,rng.integers(3))
    ar/=max(1.,np.abs(ar).sum()/.8)
    ma/=max(1.,np.abs(ma).sum()/.8)
    integrated=bool(rng.integers(2))
    e=rng.normal(0,rng.uniform(.01,.05)/(np.sqrt(duration) if integrated else 1),len(t))
    st=lfilter(np.r_[1,ma],np.r_[1,-ar],e)
    return (1-weight)*det+weight*(np.cumsum(st) if integrated else st)


def kernel_cov(rng,knots,duration):
    dist=np.abs(knots[:,None]-knots)
    u=np.clip(knots/(duration-1),0,1)
    cov=None
    for _ in range(rng.integers(1,5)):
        i=rng.integers(7)
        ell=logu(rng,2,duration/2)
        if i==0: k=np.exp(-.5*(dist/ell)**2)
        elif i==1: k=np.exp(-2*(np.sin(np.pi*dist/logu(rng,2,duration/2))/ell)**2)
        elif i==2:
            alpha=rng.uniform(.25,4)
            k=(1+(dist/ell)**2/(2*alpha))**(-alpha)
        elif i==3:
            x=np.sqrt(3)*dist/ell
            k=(1+x)*np.exp(-x)
        elif i==4: k=np.ones_like(dist)
        elif i==5: k=np.eye(len(knots))
        else: k=1+np.outer(u,u)
        k/=max(np.diag(k).max(),1e-12)
        if cov is None: cov=k
        else: cov=cov+k if rng.integers(2) else cov*k
        cov/=max(np.diag(cov).max(),1e-12)
    return (cov+cov.T)*.5


def gp_fields(rng,count,warm,duration,t):
    knots=np.linspace(-warm,duration-1,min(256,duration+warm))
    matrices=np.stack([kernel_cov(r,knots,duration) for r in streams(rng,count)])
    normal=rng.standard_normal((count,len(knots),1))
    device=os.environ.get("SSTER_GP_DEVICE","cpu")
    if device.startswith("cuda"):
        import torch
        if device=="cuda":
            ident=mp.current_process()._identity
            device=f"cuda:{((ident[-1]-1) if ident else 0)%torch.cuda.device_count()}"
        with torch.no_grad():
            cov=torch.as_tensor(matrices,device=device)
            eye=torch.eye(len(knots),device=device,dtype=cov.dtype)
            ch,info=torch.linalg.cholesky_ex(cov+1e-8*eye)
            bad=info.ne(0)
            if bad.any():
                ch[bad]=torch.as_tensor(np.stack([factor(x) for x in matrices[bad.cpu().numpy()]]),device=device)
            samples=(ch@torch.as_tensor(normal,device=device)).squeeze(-1).cpu().numpy()
    else:
        samples=np.stack([factor(c)@v[:,0] for c,v in zip(matrices,normal)])
    return np.stack([np.interp(t,knots,y) for y in samples],axis=1)


def factor(cov):
    eye=np.eye(len(cov))
    for power in range(-8,-3):
        try: return np.linalg.cholesky(cov+10.**power*eye)
        except np.linalg.LinAlgError:
            if power==-8 and np.linalg.eigvalsh(cov)[0]<-1e-8: raise
    raise np.linalg.LinAlgError("unfactorable composite GP covariance")


def innovations(rng,t,duration):
    law=int(rng.integers(4))
    level=int(rng.integers(4))
    scale=rng.uniform(.02+.07*level,.09+.07*level)*np.ones(len(t))
    for _ in range(rng.integers(4)):
        length=rounded(logu(rng,2,max(2,duration//4)))
        start=rng.integers(0,duration-length+1)
        scale[(t>=start)&(t<start+length)]*=logu(rng,.5,2)
    a,b,c=streams(rng,3)
    if law==0: noise=a.normal(size=len(t))
    elif law==1: noise=a.standard_t(3,len(t))/np.sqrt(3)
    elif law==2: noise=a.laplace(0,1/np.sqrt(2),len(t))
    else: noise=np.where(c.random(len(t))<.5,a.normal(size=len(t)),b.laplace(0,1/np.sqrt(2),len(t)))
    return scale*np.clip(noise,-8,8)


def admissible(spec,b,e):
    k=spec["kind"]
    if k=="none": return []
    overlapping=[i for i,p in enumerate(spec["parts"]) if k!="wavelet" or
                 (p["center"]+p["width"]>=b and p["center"]-p["width"]<=e-1)]
    if k=="wavelet" and not overlapping: return ["S04","S19"]
    result=["S01","S02","S03","S04"]
    parts=spec["parts"]
    if k=="sine":
        result+=["S05","S06","S09"]
        if spec["harmonic"] and any(n*spec["base"]<.5 and n not in {x["order"] for x in parts} for n in range(1,9)):
            result.append("S07")
        if spec["harmonic"] and len(parts)>=2: result.append("S08")
        if any(p["depth"]>0 for p in parts): result+=["S10","S11","S12"]
    elif k in ("square","triangle"):result+=["S13","S14"]
    else:
        result+=["S15","S16","S17","S18","S19"]
        if len(parts)>=2: result.append("S20")
    return result


def ratio(rng):
    return rng.uniform(.5,.8) if rng.integers(2) else rng.uniform(1.25,2)


def modified(spec,kind,t,rng,amplitude,duration):
    parts=[dict(p) for p in spec["parts"]]
    new=dict(spec,parts=parts)
    if kind=="S01": return -render(spec,t)
    if kind=="S02": return ratio(rng)*render(spec,t)
    if kind=="S03":
        limit=max((2*(p["f"]+(p["mod"] if p["depth"] else 0)) for p in parts),default=0) if spec["kind"]!="wavelet" else 0
        r=ratio(rng)
        while r<=limit:r=ratio(rng)
        return render(spec,t[0]+(t-t[0])/r)
    if kind=="S04": return render(spec,t)+abs(amplitude)*rng.standard_normal(len(t))
    indices=list(range(len(parts)))
    if kind in ("S10","S11","S12"): indices=[i for i in indices if parts[i]["depth"]>0]
    if kind in ("S15","S16","S17","S18","S20"):
        indices=[i for i in indices if parts[i]["center"]+parts[i]["width"]>=t[0] and parts[i]["center"]-parts[i]["width"]<=t[-1]]
    a=int(rng.choice(indices))
    p=parts[a]
    phase=lambda: signed(rng)*rng.uniform(np.pi/4,3*np.pi/4)
    if kind=="S05": p.update(wave=str(rng.choice(["square","triangle"])),start=0.,duty=rng.uniform(.2,.8))
    elif kind=="S06":
        shift=phase()
        for q in parts:q["phase"]+=shift
    elif kind=="S07":
        n=int(rng.choice([i for i in range(1,9) if i*spec["base"]<.5 and i not in {q["order"] for q in parts}]))
        parts.append(dict(p,wave="sine",f=spec["base"]*n,order=n,amp=rng.uniform(.2,1)/n,
                          depth=0.,phase=rng.uniform(0,2*np.pi)))
    elif kind in ("S08","S20"):parts.pop(a)
    elif kind=="S09":p["phase"]+=phase()
    elif kind=="S10":p["depth"]=rng.uniform(.65,.85) if p["depth"]<.45 else rng.uniform(.05,.2)
    elif kind=="S11":
        r=ratio(rng)
        while p["f"]+p["mod"]*r>=.5:r=ratio(rng)
        p["mod"]*=r
    elif kind=="S12":p["mphase"]+=phase()
    elif kind=="S13":p["start"]=(p["start"]+signed(rng)*rng.uniform(.1,.4))%1
    elif kind=="S14":p["duty"]=np.clip(p["duty"]*ratio(rng),.05,.95)
    elif kind=="S15":p["name"]=str(rng.choice([m for m in MOTHERS if m!=p["name"]]))
    elif kind=="S16":p["width"]*=ratio(rng)
    elif kind=="S17":p["center"]+=signed(rng)*rng.uniform(.25,.75)*p["width"]
    elif kind=="S18":p["amp"]*=ratio(rng)
    elif kind=="S19":parts.append(new_atom(rng,duration,int(t[0]),int(t[-1])+1,len(t)))
    else:raise ValueError(kind)
    return render(new,t)


@dataclass
class SeasonMetadata:
    components: list

    def eligible(self,i,start,stop):
        return [] if stop-start<8 else admissible(self.components[i],start,stop)


def normal_seasons(rng,N,Wwarm,Tstar):
    sr=streams(rng,5)[0]
    return SeasonMetadata([season_params(r,Tstar) for r in streams(sr,N)])


@dataclass
class Background:
    b: np.ndarray
    components: list
    trends: np.ndarray
    gp: np.ndarray
    common: np.ndarray
    seasonal: np.ndarray
    warm: int
    duration: int

    def eligible(self,i,start,stop):
        if stop-start<8:return []
        return admissible(self.components[i],start,stop)

    def season_delta(self,i,kind,start,stop,rng,amplitude):
        if kind not in self.eligible(i,start,stop):raise ValueError(f"incompatible seasonal event {kind}")
        t=np.arange(start,stop)
        ix=t+self.warm
        s=modified(self.components[i],kind,t,rng,amplitude,self.duration)
        def drive(season):
            return np.clip(np.clip(np.clip(self.trends[ix,i]+season,-2,2)+self.gp[ix,i],-4,4)+self.common[ix,i],-4,4)
        return drive(s)-drive(self.seasonal[ix,i])


def normal_background(rng,N,Wwarm,Tstar,offset=0,extra=0):
    if offset!=0: raise ValueError("record clock must begin at ordinary tick zero")
    t=np.arange(-Wwarm,Tstar+extra)
    sr,gr,cr,tr,nr=streams(rng,5)
    components=[season_params(r,Tstar) for r in streams(sr,N)]
    seasonal=np.stack([render(spec,t) for spec in components],axis=1)
    trends=np.stack([trend(r,t,Tstar) for r in streams(tr,N)],axis=1)
    gp=2*np.tanh(gp_fields(gr,N,Wwarm,Tstar,t))
    common=np.zeros_like(gp)
    if N>=2 and cr.integers(2):
        selected=cr.choice(N,cr.integers(2,N+1),replace=False)
        loads=cr.choice([-1.,1.],len(selected))*cr.uniform(.1,1,len(selected))
        common[:,selected]=np.tanh(gp_fields(cr,1,Wwarm,Tstar,t))*loads
    noise=np.stack([innovations(r,t,Tstar) for r in streams(nr,N)],axis=1)
    b=np.clip(np.clip(np.clip(trends+seasonal,-2,2)+gp,-4,4)+common,-4,4)+noise
    return Background(b,components,trends,gp,common,seasonal,Wwarm,Tstar)
