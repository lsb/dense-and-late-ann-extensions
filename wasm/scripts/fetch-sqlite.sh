#!/usr/bin/env bash
# Produce the pinned SQLite amalgamation (sqlite3.c, sqlite3.h, sqlite3ext.h,
# shell.c) in $1 (default build/sqlite).
#
# sqlite.org is not reachable from the development container, so the source
# comes from the official GitHub mirror at a release tag. SQLite >= 3.48 builds
# its amalgamation with a bundled JimTcl, so no system tclsh is needed. The
# generated files are checked against the SHA-256 values recorded below (the
# amalgamation is deterministic for a given source tree).
set -euo pipefail

VERSION=3.53.4
TAG=version-$VERSION
COMMIT=b09c88c14082339b66c7b7158d609a771e64ca69
declare -A SHA=(
  [sqlite3.c]=b1dd5d74ec7f29055a6684fa06fb3c2f6821c87dd38f9a458dfd2e8a1db28189
  [sqlite3.h]=919e7f2e8ed1d8f56ac17b412b8971c76aa5d1a879752cc6058f75e7d5910e1d
  [sqlite3ext.h]=ac9645e5c9ff0cf176efdd6e75cb5e98f46295d38e02db5c4d208826a39ab4be
  [shell.c]=8011ed018aa12969f93573b7bb1eae2d939d64d0f451b297ff847a0211c85179
)

OUT=${1:-build/sqlite}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
if [ -f "$OUT/.ok-$VERSION" ]; then exit 0; fi

SRC=${SQLITE_SRC:-$OUT/src}
if [ ! -d "$SRC/.git" ]; then
  git clone --quiet --depth 1 --branch "$TAG" https://github.com/sqlite/sqlite "$SRC"
fi
got=$(git -C "$SRC" rev-parse HEAD)
if [ "$got" != "$COMMIT" ]; then
  echo "fetch-sqlite: $SRC is at $got, expected $COMMIT ($TAG)" >&2; exit 1
fi

mkdir -p "$SRC/bld"
( cd "$SRC/bld" && ../configure --disable-tcl >/dev/null && make -s sqlite3.c shell.c >/dev/null )
for f in sqlite3.c sqlite3.h sqlite3ext.h shell.c; do
  cp "$SRC/bld/$f" "$OUT/$f"
  h=$(sha256sum "$OUT/$f" | cut -d' ' -f1)
  if [ "$h" != "${SHA[$f]}" ]; then
    echo "fetch-sqlite: SHA-256 mismatch for $f: $h" >&2; exit 1
  fi
done
touch "$OUT/.ok-$VERSION"
echo "fetch-sqlite: SQLite $VERSION amalgamation in $OUT"
