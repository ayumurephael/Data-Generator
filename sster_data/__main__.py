import argparse
from contextlib import nullcontext
import json
import multiprocessing
from pathlib import Path

from .corpus import WorkerDeviceError, finalize, generate
from .round1 import EDITION
from .storage import lock, read_json
from .replay import replay


def parse_devices(text):
    fields = [field.strip() for field in text.split(",")]
    if any(not field.isascii() or not field.isdecimal() for field in fields):
        raise argparse.ArgumentTypeError("expected comma-separated nonnegative local CUDA indices (e.g. 0,1,1)")
    return tuple(map(int,fields))


def main():
    parser = argparse.ArgumentParser(description="Build the SSTER direct-state synthetic corpus.")
    sub = parser.add_subparsers(dest="cmd",required=True)
    build = sub.add_parser("generate",help="Generate or resume a complete production split.")
    build.add_argument("--training-edition",choices=[EDITION],help="Use the 1.68M-window first-round train schedule.")
    build.add_argument("--parent-source",type=Path,help="Reuse verified full-schedule training archives or render missing tickets.")
    build.add_argument("--extra-build",type=Path,action="append",default=[],
                       help="Additional parent physical staging directory. Requires --parent-source.")
    smoke = sub.add_parser("smoke",help="Generate a small integration fixture in its own output directory.")
    check = sub.add_parser("verify",help="Rehash every shard and re-audit published labels and quotas.")
    sample = sub.add_parser("replay",help="Reproduce a stored example and compare all arrays.")
    sample.add_argument("--out",type=Path,required=True)
    sample.add_argument("--split",choices=["train","validation","test"],required=True)
    sample.add_argument("--shard",required=True,help="Shard path or filename from one manifest row.")
    sample.add_argument("--id",required=True,help="Opaque ID from the same manifest row.")
    sample.add_argument("--save",type=Path,help="Optional NPZ containing regenerated unpermuted physical traces.")
    diag = sub.add_parser("diagnose",help="Compare 256-knot and dense GP realizations; audit normal priors.")
    diag.add_argument("--out",type=Path,required=True)
    diag.add_argument("--split",choices=["train","validation","test"],default="train")
    diag.add_argument("--repetitions",type=int,default=64)
    diag.add_argument("--length",type=int,default=512)
    for p in (build,smoke,check):
        p.add_argument("--out",type=Path,required=True)
        p.add_argument("--split",choices=["train","validation","test","all"],default="train")
    for p in (build,smoke):
        p.add_argument("--workers",type=int,default=1)
        p.add_argument("--worker-devices",type=parse_devices,metavar="I,J,...",
                       help="Assign GP workers cyclically to local CUDA indices; repeat an index for more workers.")
    check.add_argument("--deep",action="store_true",help="Also decode every indexed public member using random-access offsets.")
    check.add_argument("--workers",type=int,default=1,help="CPU processes for deterministic structural ancestor checks.")
    args = parser.parse_args()
    if args.cmd == "generate":
        if (args.parent_source and not args.training_edition) or (args.extra_build and not args.parent_source):
            parser.error("parent sources require a first-round training edition; extra builds require a parent source")
        if args.training_edition and args.split not in ("train","all"):
            parser.error("the training edition applies to train only")
    if args.cmd == "diagnose":
        from .diagnostics import diagnose
        if args.repetitions < 1 or args.length <= 256:
            parser.error("diagnostics require positive repetitions and length > 256")
        print(diagnose(args.out,args.split,args.repetitions,args.length))
        return
    if args.cmd == "replay":
        print(replay(args.out/args.split,args.shard,args.id,args.save))
        return
    if args.cmd in ("generate","smoke","verify") and args.workers < 1:
        parser.error("--workers must be positive")
    for split in (["train","validation","test"] if args.split=="all" else [args.split]):
        if args.cmd == "verify":
            root = args.out/split
            with lock(root/"build.lock"), (multiprocessing.get_context("spawn").Pool(args.workers)
                                           if args.workers > 1 else nullcontext()) as workers:
                marker = read_json(root/"_SUCCESS")
                report = finalize(root,split,marker["smoke"],deep=args.deep,pool=workers)
                print(f"{split}: {report['windows']} windows, manifest SHA-256 {report['manifest_sha256']}")
        else:
            try:
                first = args.cmd == "generate" and split == "train" and args.training_edition
                generate(args.out,split,args.workers,smoke=args.cmd=="smoke",worker_devices=args.worker_devices,
                         training_edition=first or None,parent_source=args.parent_source if first else None,
                         extra_builds=args.extra_build if first else ())
            except WorkerDeviceError as exc:
                parser.error(str(exc))


if __name__ == "__main__":
    main()
