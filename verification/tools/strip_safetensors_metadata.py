#!/usr/bin/env python3
"""Remove the safetensors ``__metadata__`` block in place.

The safetensors header is an 8-byte little-endian length followed by a JSON
object; tensor data starts immediately after the header. Removing metadata
would normally shift every data offset, so this rewrites the JSON compactly and
pads it with spaces to the original header length. The data section is not
touched.

If the file has multiple hard links (setup.sh links the development
safetensors from the investigation tree), a private reflink copy is made first
so the shared source is not modified.
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile


def read_header(path):
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: truncated header length")
        n = struct.unpack("<Q", raw_len)[0]
        raw = f.read(n)
        if len(raw) != n:
            raise ValueError(f"{path}: truncated header")
    return n, json.loads(raw)


def break_hardlink(path):
    """Give ``path`` a private inode if it is hard-linked. Returns True if so."""
    if os.stat(path).st_nlink <= 1:
        return False
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".strip-",
                               dir=directory)
    os.close(fd)
    try:
        # Reflink if the filesystem supports it (btrfs/xfs); otherwise copy.
        rc = subprocess.run(["cp", "--reflink=auto", "-p", path, tmp]).returncode
        if rc != 0:
            shutil.copy2(path, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return True


def strip(path):
    n, header = read_header(path)
    if "__metadata__" not in header:
        print(f"{path}: no __metadata__")
        return

    del header["__metadata__"]
    new = json.dumps(header, separators=(",", ":")).encode()
    if len(new) > n:
        raise ValueError(f"{path}: rewritten header {len(new)} > {n}")
    new += b" " * (n - len(new))

    with open(path, "r+b") as f:
        f.seek(0)
        f.write(struct.pack("<Q", n))
        f.write(new)
    print(f"{path}: stripped __metadata__")


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} FILE.safetensors", file=sys.stderr)
        return 2
    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"{path}: not a file", file=sys.stderr)
        return 2
    _, header = read_header(path)
    if "__metadata__" not in header:
        print(f"{path}: no __metadata__")
        return 0
    if break_hardlink(path):
        print(f"{path}: broke hard link with a private copy")
    strip(path)


if __name__ == "__main__":
    sys.exit(main())
