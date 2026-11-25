"""Explicit worker placement is scoped to a split's pool and overrides only the GP device."""
import argparse
import multiprocessing
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sster_data.__main__ import main, parse_devices
from sster_data.corpus import configure_worker


@pytest.mark.parametrize("text", ["", "-1", "0,", "0,,2", "cuda:2", "1.5"])
def test_device_list_syntax(text):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_devices(text)


def test_weighted_mapping_and_no_option(monkeypatch):
    cuda = Mock()
    cuda.device_count.return_value = 3
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=cuda))
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","SSTER_GP_DEVICE"):
        monkeypatch.setenv(name,"cpu")
    # A separate process identity from previous pools must not shift the first explicit slot.
    monkeypatch.setattr(multiprocessing,"current_process",lambda:SimpleNamespace(_identity=(7,)))
    devices = parse_devices("2, 2,0")
    ctx = multiprocessing.get_context("spawn")
    for _ in range(2):
        counter = ctx.Value("q",0)
        for expected in (2,2,0,2,2):
            configure_worker(devices,counter)
            assert os.environ["SSTER_GP_DEVICE"] == f"cuda:{expected}"
            cuda.set_device.assert_called_with(expected)
    cuda.set_device.reset_mock()
    monkeypatch.setenv("SSTER_GP_DEVICE","cpu")
    configure_worker()
    cuda.set_device.assert_not_called()
    monkeypatch.setenv("SSTER_GP_DEVICE","cuda")
    configure_worker()
    assert os.environ["SSTER_GP_DEVICE"] == "cuda"
    cuda.set_device.assert_called_once_with(0)


@pytest.mark.parametrize("available,devices,reason", [(True,"0,3","outside"), (False,"0","no CUDA device")])
def test_cli_rejects_device_before_creating_pool_or_output(tmp_path,monkeypatch,capsys,available,devices,reason):
    cuda = Mock()
    cuda.is_available.return_value = available
    cuda.device_count.return_value = 3
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(multiprocessing,"get_context",lambda *_:pytest.fail("started a pool"))
    out = tmp_path/"corpus"
    monkeypatch.setattr(sys,"argv",["sster_data","smoke","--out",str(out),"--worker-devices",devices])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2 and reason in capsys.readouterr().err
    assert not out.exists()
