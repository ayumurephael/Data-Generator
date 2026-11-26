from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from sster_data import corpus
from sster_data.storage import digest, read_json, write_json


def _serial_receipts(root, n, starts, readers):
    """Original sequential archive validation, independent of the prefetch implementation."""
    for begin in starts:
        receipt = read_json(corpus.paths(root,n,begin)[-1])
        for kind in ("public","private"):
            name = receipt[kind]
            if digest(root/name) != receipt[kind+"_sha256"]:
                raise IOError(f"archive hash mismatch {name}")
        yield begin,receipt


def test_frozen_release_is_byte_identical_and_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("SSTER_GP_DEVICE","cpu")
    # More than four independent shard pairs per width, with real tar payloads.
    monkeypatch.setattr(corpus,"BATCH",2)
    with monkeypatch.context() as serial:
        serial.setattr(corpus,"_ordered_receipts",_serial_receipts)
        corpus.generate(tmp_path,"test",2,smoke=True)

    root = tmp_path/"test"
    outputs = ["manifest.jsonl","audit.json","_SUCCESS"]
    expected = {name:(root/name).read_bytes() for name in outputs}
    assert read_json(root/"_SUCCESS")["windows"] == 48
    corpus.finalize(root,"test",True,deep=True)
    assert {name:(root/name).read_bytes() for name in outputs} == expected
    # An already published split and a fresh split must both retain old failure semantics.
    receipt_path=corpus.paths(root,1,4)[-1]
    receipt=read_json(receipt_path)
    for kind in ("public","private"):
        archive=root/receipt[kind]
        original=archive.read_bytes()
        archive.write_bytes(original[:-1]+bytes([original[-1]^1]))
        with pytest.raises(OSError,match="archive hash mismatch "+receipt[kind]):
            corpus.finalize(root,"test",True)
        assert {name:(root/name).read_bytes() for name in outputs} == expected
        archive.write_bytes(original)
    for name in outputs:
        (root/name).unlink()
    # Earlier row corruption takes precedence over a later ready archive error.
    early=corpus.paths(root,1,0)[-1]
    doc=read_json(early)
    doc["rows"][0]["id"]="0"*32
    write_json(early,doc)
    (root/receipt["public"]).unlink()
    with pytest.raises(ValueError,match="receipt row disagrees"):
        corpus.finalize(root,"test",True)
    assert all(not (root/name).exists() for name in outputs)


def test_forced_out_of_order_completion_is_ordered_and_bounded(monkeypatch):
    later_ready = Event()
    consumed = []

    def starts():
        for i in range(20):
            consumed.append(i)
            yield i

    def delayed_receipt(root,n,begin):
        if begin == 0:
            assert later_ready.wait(5),"no simultaneous reader"
        if begin == corpus.FINALIZE_READERS-1:
            later_ready.set()
        return begin,{}

    monkeypatch.setattr(corpus,"_verified_receipt",delayed_receipt)
    with ThreadPoolExecutor(max_workers=corpus.FINALIZE_READERS) as readers:
        stream=corpus._ordered_receipts(None,1,starts(),readers)
        assert next(stream)[0] == 0
        assert len(consumed) == corpus.FINALIZE_READERS
        assert [i for i,_ in stream] == list(range(1,20))
