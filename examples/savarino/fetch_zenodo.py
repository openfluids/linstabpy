#!/usr/bin/env python3
"""Download Zenodo artifacts for the Savarino case-study reproduction.

The Zenodo record is open access but carries no redistribution license, so
the repository ships no data from it; this script downloads what a run needs.

  baseflow.
                   Download the published Tecplot base flow (~10 MB) into
                   ``reference/baseflow/``; the md5 is checked against the
                   Zenodo record.
  modes --f F --beta B.
                   Download forcing/response mode Tecplot files for one point
                   into ``reference/zenodo/modes/`` (HTTP range requests into
                   the Zenodo zip). Requires ``remotezip``.

Usage:
    python fetch_zenodo.py baseflow
    python fetch_zenodo.py modes --f 2 --beta 45
"""

from __future__ import annotations

import argparse

from reference_paths import BASEFLOW_DAT, BASEFLOW_NAME, zenodo_modes_dir

RECORD = "18854407"
API_BASE = f"https://zenodo.org/api/records/{RECORD}/files"

# md5 of the base flow file as reported by the Zenodo files API for the record.
BASEFLOW_MD5 = "5b3ed38fbe087633e4550b1e1231dc10"

MODES_ZIP = "resolvent_angle308_T157_Mach215_mesh300_y300_xend195.zip"
ZIP_PREFIX = "resolvent_angle308_T157_Mach215_mesh300_y300_xend195/"


def cmd_baseflow(args: argparse.Namespace) -> int:
    import hashlib
    import urllib.request

    dest = BASEFLOW_DAT
    if dest.exists():
        md5 = hashlib.md5(dest.read_bytes()).hexdigest()
        if md5 == BASEFLOW_MD5:
            print(f"already present: {dest}")
            return 0
        raise SystemExit(
            f"{dest} exists but its md5 {md5} does not match the Zenodo record "
            f"({BASEFLOW_MD5}); move it away and rerun"
        )
    url = f"{API_BASE}/{BASEFLOW_NAME}/content"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, part)
    md5 = hashlib.md5(part.read_bytes()).hexdigest()
    if md5 != BASEFLOW_MD5:
        part.unlink()
        raise SystemExit(f"download md5 {md5} != expected {BASEFLOW_MD5}")
    part.rename(dest)
    print(f"ok: {dest} (md5 verified)")
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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_base = sub.add_parser(
        "baseflow", help="download the published Tecplot base flow (md5-checked)"
    )
    p_base.set_defaults(func=cmd_baseflow)

    p_modes = sub.add_parser(
        "modes", help="download forcing/response modes for one (f, beta) point"
    )
    p_modes.add_argument(
        "--f",
        required=True,
        help="frequency, formatted as in the Zenodo filenames (e.g. 2, 0.0001)",
    )
    p_modes.add_argument(
        "--beta",
        required=True,
        help="spanwise wavenumber, formatted as in the Zenodo filenames (e.g. 45)",
    )
    p_modes.set_defaults(func=cmd_modes)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
