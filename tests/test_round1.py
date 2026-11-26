import numpy as np
import pytest

from sster_data import round1
from sster_data.scm import WIDTHS
from sster_data.storage import write_json


def test_quota_contract():
    table = round1.quotas()
    assert sum(table.values()) == 1680000
    mode_counts = [sum(m for (_, _, mode), m in table.items() if mode == name)
                   for name in ("healthy","process","observation","mixed")]
    assert mode_counts == [504000,588000,470400,117600]
    for k in range(1,7):
        assert sum(m for (_, k1, _), m in table.items() if k1 == k) == 196000
    for width in WIDTHS:
        cells = round1.target_counts(width)
        assert sum(cells.values()) == 240000 and min(cells.values()) > 0
        raw = sum(k*m for (n, k, _), m in table.items() if n == width)
        assert raw % width == 0
    assert table[64,1,"process"] - 40*round1.FOCAL == 328


def test_flow_respects_joint_margins_and_randomizes_reproducibly():
    groups = np.array([[0,0,0,1], [1,1,2,2], [0,2,2,2], [0,1,2,2]]*9)
    demand = [19,23,30]
    def pick(seed):
        return round1.select_tickets(groups.ravel(),demand,4,2,np.random.default_rng(seed))
    ids = pick(1)
    np.testing.assert_array_equal(ids,pick(1))
    assert not np.array_equal(ids,pick(2))
    np.testing.assert_array_equal(np.bincount(groups.ravel()[ids]),demand)
    assert np.all(np.bincount(ids//4)==2)
    assert np.all(np.diff(ids)>0)
    with pytest.raises(ValueError,match="no feasible"):
        round1.select_tickets([0,0,0,0,1,1,1,1],[1,3],4,2,np.random.default_rng(2))


def test_edition_cannot_mix_release_directories(tmp_path):
    build = tmp_path/"build"
    build.mkdir()
    np.save(build/"plan-4.npy",np.arange(3))
    with pytest.raises(ValueError,match="existing full-size"):
        round1.bind(tmp_path,round1.EDITION)
    (build/"plan-4.npy").unlink()
    round1.bind(tmp_path,round1.EDITION)
    assert round1.is_first_round(tmp_path)
    with pytest.raises(ValueError,match="specify"):
        round1.bind(tmp_path,None)
    write_json(build/"training-edition.json",{"training_edition":"other"})
    with pytest.raises(ValueError,match="unknown"):
        round1.is_first_round(tmp_path)
