"""Build the Python package dense-late-ann (metadata is in pyproject.toml).

The two SQLite extensions (ext/dense -> denseann, ext/late -> late) are
compiled as plain loadable modules, not as Python extension modules: they are
loaded by SQLite, not imported by Python, so they have no Python ABI and the
wheel is tagged py3-none-<platform>. They are compiled against the headers of
the pinned SQLite amalgamation (wasm/scripts/fetch-sqlite.sh), which are looked
up in this order:

  1. $SQLITE_INCLUDE_DIR
  2. python/dense_late_ann/_sqlite_include/   (present in the sdist)
  3. build/sqlite/                            (after `make sqlite`)
  4. otherwise wasm/scripts/fetch-sqlite.sh is run to produce build/sqlite/

Index construction is multi-threaded (-DDENSE_ANN_THREADS, -DLATE_THREADS), as
in tools/build_db.py; queries are unaffected.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # setuptools < 70.1
    from wheel.bdist_wheel import bdist_wheel

HERE = Path(__file__).resolve().parent
PKG = "dense_late_ann"
PKG_DIR = HERE / "python" / PKG
VENDORED_INC = PKG_DIR / "_sqlite_include"
HEADERS = ("sqlite3.h", "sqlite3ext.h")

EXTENSIONS = {
    # module file name (gives SQLite's default entry point) -> (sources, define)
    "denseann": (["ext/dense/dense_ann.c", "ext/dense/hnsw.c", "ext/dense/ivf.c",
                  "ext/dense/pq.c", "ext/dense/rawpage.c"], "DENSE_ANN_THREADS"),
    "late": (["ext/late/late_codec.c", "ext/late/late_page.c", "ext/late/late_plaid.c"],
             "LATE_THREADS"),
}


def sqlite_include_dir():
    candidates = [os.environ.get("SQLITE_INCLUDE_DIR"), VENDORED_INC, HERE / "build" / "sqlite"]
    for d in candidates:
        if d and all((Path(d) / h).is_file() for h in HEADERS):
            return str(d)
    script = HERE / "wasm" / "scripts" / "fetch-sqlite.sh"
    if script.is_file():
        print("dense-late-ann: SQLite headers not found; running wasm/scripts/fetch-sqlite.sh", flush=True)
        subprocess.run(["bash", str(script), str(HERE / "build" / "sqlite")], check=True, cwd=HERE)
        return str(HERE / "build" / "sqlite")
    sys.exit("dense-late-ann: sqlite3.h/sqlite3ext.h of SQLite 3.53.4 not found; set SQLITE_INCLUDE_DIR")


class BuildLoadable(build_ext):
    """Compile each extension into <package>/<name>.so (.dylib on macOS)."""

    def get_ext_filename(self, fullname):
        suffix = ".dylib" if sys.platform == "darwin" else ".dll" if sys.platform == "win32" else ".so"
        return os.path.join(*fullname.split(".")) + suffix

    def build_extension(self, ext):
        out = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cc = self.compiler
        args = ["-O2", "-g0", "-std=c11", "-pthread"]
        objects = cc.compile(ext.sources, output_dir=self.build_temp, macros=ext.define_macros,
                             include_dirs=ext.include_dirs, extra_postargs=args, depends=ext.depends)
        cc.link_shared_object(objects, out, libraries=["m"], extra_postargs=["-pthread"],
                              target_lang="c")


class SdistWithHeaders(sdist):
    """Put the pinned SQLite headers into the sdist, so it builds offline."""

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        inc = Path(sqlite_include_dir())
        dest = Path(base_dir) / "python" / PKG / "_sqlite_include"
        dest.mkdir(parents=True, exist_ok=True)
        for h in HEADERS:
            shutil.copyfile(inc / h, dest / h)


class PlatformWheel(bdist_wheel):
    """Platform-specific, but independent of the Python version and ABI."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, plat = super().get_tag()
        return "py3", "none", plat


def extensions():
    if any(a in sys.argv for a in ("egg_info", "sdist", "dist_info")):
        inc = None  # metadata only: do not fetch SQLite
    else:
        inc = sqlite_include_dir()
    exts = []
    for name, (srcs, define) in EXTENSIONS.items():
        exts.append(Extension(
            f"{PKG}.{name}", sources=srcs,
            include_dirs=[d for d in (inc, "wasm/src") if d],
            define_macros=[(define, None), ("_GNU_SOURCE", None)],
            depends=[str(p.relative_to(HERE)) for p in (HERE / Path(srcs[0]).parent).glob("*.h")]
                    + ["wasm/src/httpvfs.h"],
        ))
    return exts


setup(
    ext_modules=extensions(),
    cmdclass={"build_ext": BuildLoadable, "sdist": SdistWithHeaders, "bdist_wheel": PlatformWheel},
)
