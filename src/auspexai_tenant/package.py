"""Deterministic executor-package archive — the upload body for `POST
/api/v0/packages` (#40a, coordinator-served provisioning, researcher leg).

`build_package_archive` packages an executor package dir as a gzipped tar whose
bytes are a pure function of the package *tree* (the same content the
`compute_package_digest` provenance pin covers): same tree in → identical
archive bytes out, regardless of file mtimes, creation order, umask, or owner.
That makes the archive content-addressable alongside the digest the coordinator
verifies (`X-Package-Digest`), and lets the coordinator dedupe re-uploads
byte-for-byte.

Canonical form:
  - file set + order: exactly `manifest._iter_package_files` — every regular
    file under the package dir, sorted by POSIX relpath, minus the digest
    exclusions (`manifest.json`, `manifest.json.sig`, `*.pyc`, `__pycache__/`).
    Sharing the digest's enumeration seam guarantees archive and digest can
    never disagree about a package's contents.
  - entries are regular files only — symlinks are dereferenced to their target
    bytes (matching the digest's content-addressed view; no link entries),
    directories are implicit, empty dirs are not represented.
  - fixed metadata: uid/gid 0, empty uname/gname, mtime 0, mode 0o644
    (the executor is invoked via the manifest's `command` argv, so the
    executable bit is not load-bearing).
  - PAX tar format inside a gzip stream with mtime 0, no embedded filename,
    fixed compression level — no wall-clock or host identity leaks into the
    bytes.
"""

from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path

from auspexai_tenant.manifest import _iter_package_files

# Pinned so the XFL byte in the gzip header (derived from the level) is stable.
_GZIP_COMPRESSLEVEL = 9


def build_package_archive(package_dir: str | Path) -> bytes:
    """Build the canonical tar.gz of the executor package staged in
    `package_dir`. Deterministic: the same tree always yields identical bytes
    (see the module docstring for the canonical form)."""
    out = io.BytesIO()
    # filename="" keeps FNAME out of the gzip header; mtime=0 fixes MTIME.
    with gzip.GzipFile(
        filename="", fileobj=out, mode="wb", compresslevel=_GZIP_COMPRESSLEVEL, mtime=0
    ) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf:
            for rel, path in _iter_package_files(package_dir):
                data = path.read_bytes()  # dereferences symlinks
                info = tarfile.TarInfo(name=rel)
                info.size = len(data)
                info.mtime = 0
                info.mode = 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                tf.addfile(info, io.BytesIO(data))
    return out.getvalue()
