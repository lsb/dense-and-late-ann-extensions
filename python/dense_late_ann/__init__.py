"""SQLite extensions for dense and late-interaction ANN search.

The wheel carries two loadable SQLite extensions compiled from ext/dense and
ext/late of the dense-and-late-ann-extensions repository:

  denseann  virtual table dense_ann   (entry point sqlite3_denseann_init)
  late      virtual table late_plaid  (entry point sqlite3_late_init)

    import sqlite3, dense_late_ann
    db = sqlite3.connect("x.db")
    dense_late_ann.load(db)                 # both extensions
    dense_late_ann.extension_path("late")   # for sqlite3's .load, or other drivers

`load()` accepts a connection of the standard sqlite3 module (if it was built
with extension loading) or of apsw.
"""
from __future__ import annotations

import sys
from pathlib import Path

__version__ = "0.1.0"
__all__ = ["EXTENSIONS", "extension_path", "load", "connect"]

_HERE = Path(__file__).resolve().parent
_SUFFIX = ".dylib" if sys.platform == "darwin" else ".dll" if sys.platform == "win32" else ".so"

#: short name -> (file stem, entry point, virtual table module)
EXTENSIONS = {
    "dense": ("denseann", "sqlite3_denseann_init", "dense_ann"),
    "late": ("late", "sqlite3_late_init", "late_plaid"),
}


def extension_path(name: str = "dense") -> str:
    """Absolute path of a loadable extension: 'dense' (dense_ann) or 'late' (late_plaid)."""
    try:
        stem = EXTENSIONS[name][0]
    except KeyError:
        raise ValueError(f"unknown extension {name!r}; expected one of {sorted(EXTENSIONS)}") from None
    p = _HERE / (stem + _SUFFIX)
    if not p.is_file():
        raise FileNotFoundError(f"{p} is missing; the package was installed without its compiled extensions")
    return str(p)


def load(conn, names=("dense", "late")):
    """Load extensions into an open sqlite3 or apsw connection; returns the connection."""
    if isinstance(names, str):
        names = (names,)
    conn.enable_load_extension(True)
    try:
        for n in names:
            # The file name gives SQLite's default entry point (denseann.so ->
            # sqlite3_denseann_init), so no entry point argument is needed.
            conn.load_extension(extension_path(n))
    finally:
        conn.enable_load_extension(False)
    return conn


def connect(path, **kwargs):
    """Open a database with the standard sqlite3 module, falling back to apsw
    when sqlite3 cannot load extensions, and load both extensions into it.
    Keyword arguments go to sqlite3.connect (apsw connections are always in
    autocommit mode, as with isolation_level=None)."""
    import sqlite3
    if hasattr(sqlite3.Connection, "enable_load_extension"):
        return load(sqlite3.connect(str(path), **kwargs))
    try:
        import apsw
    except ImportError:
        raise RuntimeError("this Python's sqlite3 module cannot load extensions; "
                           "pip install 'dense-late-ann[apsw]'") from None
    return load(apsw.Connection(str(path)))
