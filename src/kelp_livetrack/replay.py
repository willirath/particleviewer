"""kelp-live-replay: grow a copy of a finished kelp Zarr store record block by record block.

Writes in the order kelp's Zarr writer flushes (time first, then each per-record
variable is resized and filled in turn), so a reader sees the same torn
intermediate states as against a real run. For demos and for testing --follow.

    kelp-live-replay done.zarr growing.zarr --rate 2 --block 4
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

import zarr

ORDER = ("lon", "lat", "z", "status")


def per_record(src) -> list[str]:
    """Per-record (trajectory, time) variables in kelp's flush order."""
    P = src["lon"].shape[0]
    names = [n for n, a in src.arrays()
             if a.ndim == 2 and a.shape[0] == P and not n.startswith("restart_")]
    return [n for n in ORDER if n in names] + sorted(n for n in names if n not in ORDER)


def start(src_path: str, dst_path: str, force: bool = False):
    """Create the destination with all static arrays and zero records."""
    if os.path.exists(dst_path):
        if not force:
            raise SystemExit(f"{dst_path} exists (use --force to replace it)")
        shutil.rmtree(dst_path)
    src = zarr.open_group(src_path, mode="r")
    dst = zarr.open_group(dst_path, mode="w")
    dst.attrs.update(dict(src.attrs))
    grow = set(per_record(src)) | {"time"}
    for name, a in src.arrays():
        if name.startswith("restart_"):
            continue
        dims = a.metadata.dimension_names
        if name in grow:
            shape = (0,) if a.ndim == 1 else (a.shape[0], 0)
            d = dst.create_array(name, shape=shape, chunks=a.chunks, dtype=a.dtype,
                                 fill_value=a.fill_value, dimension_names=dims)
        else:
            d = dst.create_array(name, shape=a.shape, chunks=a.chunks, dtype=a.dtype,
                                 fill_value=a.fill_value, dimension_names=dims)
            d[...] = a[...]
        d.attrs.update(dict(a.attrs))
    return src, dst


def append(src, dst, k0: int, k1: int, names: list[str]) -> None:
    """Copy records [k0, k1) like kelp's _flush_block: time first, then var by var."""
    t = dst["time"]
    t.resize((k1,))
    t[k0:k1] = src["time"][k0:k1]
    for n in names:
        a = dst[n]
        a.resize((a.shape[0], k1))
        a[:, k0:k1] = src[n][:, k0:k1]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kelp-live-replay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--rate", type=float, default=2.0, help="records per second")
    ap.add_argument("--block", type=int, default=1, help="records per flush")
    ap.add_argument("--initial", type=int, default=0, help="records written immediately")
    ap.add_argument("--force", action="store_true", help="replace an existing destination")
    args = ap.parse_args(argv)

    src, dst = start(args.src, args.dst, args.force)
    names = per_record(src)
    n = src["time"].shape[0]
    k = min(args.initial, n)
    if k:
        append(src, dst, 0, k, names)
    print(f"replaying {n} records of {args.src} into {args.dst} "
          f"({args.block} per flush, {args.rate:g}/s)")
    try:
        while k < n:
            time.sleep(args.block / args.rate)
            k1 = min(n, k + args.block)
            append(src, dst, k, k1, names)
            k = k1
            print(f"\r{k}/{n}", end="", flush=True)
    except KeyboardInterrupt:
        pass
    print()


if __name__ == "__main__":
    main()
