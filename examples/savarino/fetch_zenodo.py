#!/usr/bin/env python3
"""Downloads optional Zenodo artifacts for the Savarino case-study reproduction.

The base flow Tecplot file is **shipped** under ``reference/baseflow/``.
This script fetches resolvent mode archives when regenerating shipped fixtures
(e.g. ``pack_reference_modes.py``).

  modes --f F --beta B
                   Download forcing/response mode Tecplot files for one point
                   into ``reference/zenodo/modes/`` (HTTP range requests into
                   the Zenodo zip). Requires ``remotezip``.

Usage:
    python fetch_zenodo.py modes --f 2 --beta 45
"""
from __future__ import annotations

import argparse
from pathlib import Path

from reference_paths import zenodo_modes_dir

RECORD = "18854407"
API_BASE = f"https://zenodo.org/api/records/{RECORD}/files"

MODES_ZIP = "resolvent_angle308_T157_Mach215_mesh300_y300_xend195.zip"
ZIP_PREFIX = "resolvent_angle308_T157_Mach215_mesh300_y300_xend195/"


def cmd_modes(args: argparse.Namespace) -> int:
    try:
        from remotezip import RemoteZip
    except ImportError:
        raise SystemExit(
            "modes requires the 'remotezip' package (HTTP range requests into "
            "the Zenodo zip without downloading it whole): pip install remotezip"
        )

    zip_url = f"{API_BASE}/{MODES_ZIP}/content"
    out_dir = zenodo_modes_dir()
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

    p_modes = sub.add_parser("modes", help="download forcing/response modes for one (f, beta) point")
    p_modes.add_argument("--f", required=True, help="frequency, formatted as in the Zenodo filenames (e.g. 2, 0.0001)")
    p_modes.add_argument("--beta", required=True, help="spanwise wavenumber, formatted as in the Zenodo filenames (e.g. 45)")
    p_modes.set_defaults(func=cmd_modes)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
