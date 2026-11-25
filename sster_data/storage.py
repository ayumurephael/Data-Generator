"""Deterministic archives, offset indexing, receipts and atomic publication."""
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import zipfile

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(4 << 20), b""):
            h.update(part)
    return h.hexdigest()


def encode(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",",":"), allow_nan=False).encode()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
    with temp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp,path)


def write_json(path, data):
    atomic(path,encode(data)+b"\n")


def read_json(path):
    with open(path,"rb") as f:
        return json.load(f)


def pack(arrays, level=1):
    """np.load-compatible ZIP without platform timestamps or pickled arrays."""
    out = io.BytesIO()
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=level) as z:
        for key,array in arrays.items():
            payload = io.BytesIO()
            np.lib.format.write_array(payload,np.asarray(array),allow_pickle=False)
            z.writestr(zipfile.ZipInfo(f"{key}.npy",(1980,1,1,0,0,0)),payload.getvalue(),
                       compress_type=zipfile.ZIP_DEFLATED,compresslevel=level)
    return out.getvalue()


def unpack(data):
    with np.load(io.BytesIO(data),allow_pickle=False) as f:
        return {name: f[name] for name in f.files}


class Archive:
    def __init__(self, path):
        self.path = Path(path)
        self.temp = self.path.with_suffix(f".{os.getpid()}.tmp")
        self.file = self.temp.open("wb")
        self.tar = tarfile.open(fileobj=self.file,mode="w",format=tarfile.USTAR_FORMAT)
        self.rows = []

    def add(self,name,data):
        start = self.tar.offset + 512
        item = tarfile.TarInfo(name)
        item.size = len(data)
        self.tar.addfile(item,io.BytesIO(data))
        return start,len(data)

    def __enter__(self):
        return self

    def __exit__(self, typ, exc, tb):
        try:
            self.tar.close()
            self.file.flush()
            os.fsync(self.file.fileno())
        finally:
            self.file.close()
        if typ is None:
            os.replace(self.temp,self.path)


@contextmanager
def lock(path):
    """Release advisory process lock on exceptions and OS process death."""
    with open(path, "a+b") as f:
        if os.name == "nt":
            import msvcrt
            if f.tell() == 0:
                f.write(b"\0")
                f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
