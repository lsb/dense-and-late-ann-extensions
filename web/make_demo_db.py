#!/usr/bin/env python3
"""Build (or copy) the demo database build/web/<corpus>.db.

The demo uses the benchmark matrix's database format: tools/build_db.py puts
docs + FTS5 + dense_ann (graph and IVF) + late_plaid (warp + plaid) into one
file, VACUUMs it and finalizes the page hints. If build/matrix/<corpus>.db is
already complete and was built from the current extension sources (its
manifest's SHA-256 digests match), it is copied; otherwise the database is
built with tools/build_db.py --out-dir build/web.

  python3 web/make_demo_db.py [words-10k] [--rebuild]
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "build" / "web"
MATRIX = REPO / "build" / "matrix"


def current_digests():
    sys.path.insert(0, str(REPO / "tools"))
    import build_db  # noqa: E402
    return {k: build_db.ext_digest(k) for k in build_db.EXT_SOURCES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?", default="words-10k")
    ap.add_argument("--rebuild", action="store_true", help="always build, never copy")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    src, man = MATRIX / f"{a.corpus}.db", MATRIX / f"{a.corpus}.db.json"
    dst = OUT / f"{a.corpus}.db"
    if not a.rebuild and src.exists() and man.exists() and not Path(str(src) + "-journal").exists():
        m = json.loads(man.read_text())
        if m.get("ext_sources_sha256") == current_digests() and m.get("bytes") == src.stat().st_size:
            shutil.copyfile(src, dst)
            shutil.copyfile(man, str(dst) + ".json")
            print(f"copied {src} -> {dst} ({dst.stat().st_size / 2**20:.1f} MiB)")
            return
        print(f"{src} is stale or incomplete; building")
    subprocess.run([sys.executable, str(REPO / "tools" / "build_db.py"), a.corpus, "--out-dir", str(OUT)], check=True)


if __name__ == "__main__":
    main()
