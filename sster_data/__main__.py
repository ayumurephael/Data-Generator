import argparse
import json
from pathlib import Path

from .corpus import finalize, generate
from .storage import lock, read_json
from .replay import replay


def main():
    parser = argparse.ArgumentParser(description="Build the SSTER direct-state synthetic corpus.")
    sub = parser.add_subparsers(dest="cmd",required=True)
    build = sub.add_parser("generate",help="Generate or resume a complete production split.")
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
    check.add_argument("--deep",action="store_true",help="Also decode every indexed public member using random-access offsets.")
    args = parser.parse_args()
    if args.cmd == "diagnose":
        from .diagnostics import diagnose
        if args.repetitions < 1 or args.length <= 256:
            parser.error("diagnostics require positive repetitions and length > 256")
        print(diagnose(args.out,args.split,args.repetitions,args.length))
        return
    if args.cmd == "replay":
        print(replay(args.out/args.split,args.shard,args.id,args.save))
        return
    if args.cmd in ("generate","smoke") and args.workers < 1:
        parser.error("--workers must be positive")
    for split in (["train","validation","test"] if args.split=="all" else [args.split]):
        if args.cmd == "verify":
            root = args.out/split
            with lock(root/"build.lock"):
                marker = read_json(root/"_SUCCESS")
                report = finalize(root,split,marker["smoke"],deep=args.deep)
                print(f"{split}: {report['windows']} windows, manifest SHA-256 {report['manifest_sha256']}")
        else:
            generate(args.out,split,args.workers,smoke=args.cmd=="smoke")


if __name__ == "__main__":
    main()
