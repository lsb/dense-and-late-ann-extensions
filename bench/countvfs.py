"""An APSW VFS shim that records every read of the main database file.

Used to estimate httpvfs costs natively: open a fresh connection per query
(cold page cache), run it, and look at the reads. Reads are recorded as
(offset, length) pairs; ``pages`` counts distinct pages touched.
"""
import apsw


class CountingVFS(apsw.VFS):
    def __init__(self, name="counting", base=""):
        self.name = name
        self.reads = []
        super().__init__(name, base)

    def xOpen(self, name, flags):
        return CountingFile(self, "", name, flags)

    def reset(self):
        self.reads = []


class CountingFile(apsw.VFSFile):
    def __init__(self, vfs, inheritfromvfsname, filename, flags):
        self.vfs = vfs
        fname = filename.filename() if isinstance(filename, apsw.URIFilename) else filename
        self.is_main = bool(fname) and not fname.endswith(("-journal", "-wal", "-shm"))
        super().__init__(inheritfromvfsname, filename, flags)

    def xRead(self, amount, offset):
        if self.is_main:
            self.vfs.reads.append((offset, amount))
        return super().xRead(amount, offset)


def pages_touched(reads, page_size):
    return len({off // page_size for off, _ in reads})
