"""Numerical contracts for all 40 perturbations, seasonal metadata, and replay."""
import numpy as np
import pytest

from sster_data import catalog as cat


@pytest.mark.parametrize("i",range(1,21))
def test_local_minimum_and_sign(i):
    name=f"L{i:02}"
    d=cat.MIN_LENGTH[name]
    a=cat.local_wave(name,d,np.random.default_rng(31),1.)
    b=cat.local_wave(name,d,np.random.default_rng(31),3.)
    assert a.shape==(d,) and np.isfinite(a).all() and np.max(abs(a))>0
    np.testing.assert_allclose(b,3*a)
    positives={1,3,5,8,10,12,13,18}
    negatives={2,4,6,9,11,14,15,19}
    if i in positives:
        assert np.all(a>=0)
    if i in negatives:
        assert np.all(a<=0)
    if i==16 or i==17:
        assert a[np.argmax(abs(a[:d//2]))]*a[-1]<0
    with pytest.raises(ValueError):
        cat.local_wave(name,d-1,np.random.default_rng(4),1.)


def test_every_seasonal_template_is_eligible_and_replayable():
    rng=np.random.default_rng(74191)
    todo={f"S{i:02}" for i in range(1,21)}
    t=np.arange(512)
    for _ in range(1500):
        spec=cat.season_params(rng,len(t))
        for name in todo.intersection(cat.admissible(spec,0,len(t))):
            base=cat.render(spec,t)
            changed=cat.modified(spec,name,t,np.random.default_rng(123),1.2,len(t))
            np.testing.assert_array_equal(changed,cat.modified(spec,name,t,np.random.default_rng(123),1.2,len(t)))
            np.testing.assert_array_equal(base,cat.render(spec,t))
            assert changed.shape==base.shape and np.isfinite(changed).all() and not np.array_equal(base,changed)
            todo.remove(name)
        if not todo:
            break
    assert not todo


def test_independent_metadata_and_guard_continuation():
    def draw(extra):
        return cat.normal_background(np.random.default_rng(47),4,32,128,extra=extra)
    base=draw(0)
    extended=draw(13)
    assert cat.normal_seasons(np.random.default_rng(47),4,32,128).components==base.components
    for k in ("b","trends","seasonal","gp","common"):
        np.testing.assert_allclose(getattr(base,k),getattr(extended,k)[:160],rtol=0,atol=0)
    for i in range(4):
        available=base.eligible(i,0,50)
        for k in available:
            d=base.season_delta(i,k,0,50,np.random.default_rng(77),.4)
            assert d.shape==(50,) and np.isfinite(d).all()


def test_nyquist_mothers_and_replay_eigenfactor():
    r=np.random.default_rng(555)
    for _ in range(750):
        spec=cat.season_params(r,256)
        for p in spec["parts"] if spec["kind"] not in ("none","wavelet") else []:
            assert 0<p["f"]<.5 and (not p["depth"] or p["f"]+p["mod"]<.5)
    for name in cat.MOTHERS:
        x,y=cat.mother(name)
        assert x[0]==-1 and x[-1]==1 and np.max(np.abs(y))==1
    knots=np.linspace(-5,36,24)
    for _ in range(12):
        cov=cat.kernel_cov(r,knots,37)
        factor=cat.factor(cov)
        np.testing.assert_allclose(factor@factor.T,cov+1e-8*np.eye(24),atol=1e-12)
