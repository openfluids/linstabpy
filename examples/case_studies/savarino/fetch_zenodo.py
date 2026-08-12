#!/usr/bin/env python3
"""Downloads the reference dataset for the Savarino case-study reproduction.

Record 10.5281/zenodo.18854407 on zenodo.org holds the published base flow,
forcing/response modes, and eigenmodes. Two subcommands:

  baseflow        Download the base-flow file into $SAVARINO_DATA (or
                   data/ next to this script).
  modes --f F --beta B
                   Download the four mode files (forcing/response, real and
                   imaginary parts) for a given (f, beta) point from inside
                   the resolvent-modes zip archive, using HTTP range
                   requests so only the needed members are transferred.
                   Requires the ``remotezip`` package: pip install remotezip

Usage:
    python fetch_zenodo.py baseflow
    python fetch_zenodo.py modes --f 2 --beta 45
"""
from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

RECORD = "18854407"
API_BASE = f"https://zenodo.org/api/records/{RECORD}/files"

BASEFLOW_NAME = "state_atcenter_mesh300_y300_xend195_angle308_T157_Mach215.dat"
MODES_ZIP = "resolvent_angle308_T157_Mach215_mesh300_y300_xend195.zip"
ZIP_PREFIX = "resolvent_angle308_T157_Mach215_mesh300_y300_xend195/"

DATA_DIR = Path(os.environ.get("SAVARINO_DATA", Path(__file__).resolve().parent / "data"))


def _download(name: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{API_BASE}/{name}/content"
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def cmd_baseflow(_args: argparse.Namespace) -> int:
    dest = DATA_DIR / BASEFLOW_NAME
    if dest.exists():
        print(f"already present: {dest}")
        return 0
    _download(BASEFLOW_NAME, dest)
    return 0


def cmd_modes(args: argparse.Namespace) -> int:
    try:
        from remotezip import RemoteZip
    except ImportError:
        raise SystemExit(
            "modes requires the 'remotezip' package (HTTP range requests into "
            "the Zenodo zip without downloading it whole): pip install remotezip"
        )

    zip_url = f"{API_BASE}/{MODES_ZIP}/content"
    out_dir = DATA_DIR / "modes"
    out_dir.mkdir(parents=True, exist_ok=True)

    members = [
        f"{kind}_atcenter_f{args.f}_w{args.beta}_n1{part}.dat"
        for kind in ("forcing", "response")
        for part in ("real", "imag")
    ]
    print(f"opening {zip_url} via range requests")
    with RemoteZip(zip_url) as z:
        for name in members:
            dest = out_dir / name
            if dest.exists():
                print(f"already present: {dest}")
                continue
            member_path = ZIP_PREFIX + name
            print(f"extracting {member_path} -> {dest}")
            with z.open(member_path) as src, open(dest, "wb") as fh:
                fh.write(src.read())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_base = sub.add_parser("baseflow", help="download the base-flow file")
    p_base.set_defaults(func=cmd_baseflow)

    p_modes = sub.add_parser("modes", help="download forcing/response modes for one (f, beta) point")
    p_modes.add_argument("--f", required=True, help="frequency, formatted as in the Zenodo filenames (e.g. 2, 0.0001)")
    p_modes.add_argument("--beta", required=True, help="spanwise wavenumber, formatted as in the Zenodo filenames (e.g. 45)")
    p_modes.set_defaults(func=cmd_modes)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
