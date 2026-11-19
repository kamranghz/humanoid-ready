"""Inspect and download ONE file group of ONE Nymeria sequence from the official URL list.

The URL list (``Nymeria_download_urls.json``) comes from the Nymeria download page. Its links
are signed and expire, so this script never prints a link. Nymeria is CC BY-NC 4.0.

Usage (hready environment, Windows):
    # 1. what does the list say about a sequence? (prints groups and sizes, no links)
    python scripts/nymeria_fetch.py --json PATH --sequence SEQ --inspect

    # 2. download one group, with resume and a checksum, into <out>/<sequence>/
    python scripts/nymeria_fetch.py --json PATH --sequence SEQ --group GROUP --out OUT

The download refuses to start when the file is larger than ``--max-gb`` (default 60) or when
less than ``--keep-free-gb`` (default 100) would remain free on the target drive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

URL_KEYS = ("download_url", "url", "link")
SIZE_HINTS = ("size", "bytes")
HASH_HINTS = ("sha1", "sha256", "md5", "checksum")


def find_sequence(node, sequence: str):
    """Depth-first search for the dict stored under the key ``sequence``."""
    if isinstance(node, dict):
        if sequence in node and isinstance(node[sequence], dict):
            return node[sequence]
        for value in node.values():
            found = find_sequence(value, sequence)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_sequence(value, sequence)
            if found is not None:
                return found
    return None


def redact(value):
    """Same structure with every link replaced, so nothing signed is printed."""
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value[:3]] + (
            [f"... ({len(value)} items)"] if len(value) > 3 else []
        )
    if isinstance(value, str) and value.lower().startswith(("http://", "https://")):
        return "<link removed>"
    return value


def entry_url(entry) -> str | None:
    if not isinstance(entry, dict):
        return None
    for key in URL_KEYS:
        if isinstance(entry.get(key), str):
            return entry[key]
    for value in entry.values():
        if isinstance(value, str) and value.lower().startswith(("http://", "https://")):
            return value
    return None


def entry_field(entry: dict, hints) -> tuple[str, object] | None:
    for key, value in entry.items():
        if any(hint in key.lower() for hint in hints) and not isinstance(value, dict):
            return key, value
    return None


def file_name(entry: dict, url: str) -> str:
    name = entry.get("filename") or entry.get("file_name")
    if isinstance(name, str) and name:
        return Path(name).name
    return Path(unquote(urlparse(url).path)).name


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def download(url: str, target: Path, total: int | None) -> None:
    """Download with resume into ``target.part``, then rename."""
    part = target.with_name(target.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url)
    if have:
        request.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        resumed = have and response.status == 206
        mode = "ab" if resumed else "wb"
        done = have if resumed else 0
        next_report = 0.05
        with open(part, mode) as handle:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                handle.write(block)
                done += len(block)
                if total and done / total >= next_report:
                    print(f"  {done / 1e9:.2f} / {total / 1e9:.2f} GB", flush=True)
                    next_report += 0.05
    part.replace(target)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--json", type=Path, required=True, help="Nymeria_download_urls.json"
    )
    parser.add_argument("--sequence", required=True)
    parser.add_argument(
        "--inspect", action="store_true", help="print groups and sizes, no download"
    )
    parser.add_argument("--group", help="key of the file group to download")
    parser.add_argument(
        "--out", type=Path, help="dataset root; files go to <out>/<sequence>/"
    )
    parser.add_argument("--max-gb", type=float, default=60.0)
    parser.add_argument("--keep-free-gb", type=float, default=100.0)
    args = parser.parse_args(argv)

    with open(args.json, encoding="utf-8") as handle:
        data = json.load(handle)
    sequence = find_sequence(data, args.sequence)
    if sequence is None:
        print(f"sequence not found in the list: {args.sequence}", file=sys.stderr)
        return 2

    if args.inspect or not args.group:
        print(f"sequence {args.sequence}: {len(sequence)} entries")
        for key, entry in sequence.items():
            print(f"  {key}: {json.dumps(redact(entry))}")
        return 0

    entry = sequence.get(args.group)
    if entry is None:
        close = [key for key in sequence if args.group in key]
        print(f"group not found: {args.group}; close matches: {close}", file=sys.stderr)
        return 2
    url = entry_url(entry)
    if url is None:
        print("no link in that entry; the list may have expired", file=sys.stderr)
        return 2
    if args.out is None:
        print("--out is required to download", file=sys.stderr)
        return 2

    size_field = entry_field(entry, SIZE_HINTS)
    total = int(size_field[1]) if size_field and str(size_field[1]).isdigit() else None
    hash_field = entry_field(entry, HASH_HINTS)
    target_dir = args.out / args.sequence
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / file_name(entry, url)
    free_gb = shutil.disk_usage(target_dir).free / 1e9
    print(
        f"file {target.name}; size {total / 1e9:.2f} GB"
        if total
        else f"file {target.name}; size unknown"
    )
    print(f"free on the drive: {free_gb:.1f} GB")
    if total is None:
        print(
            "size not in the list: refusing; pass the size or inspect first",
            file=sys.stderr,
        )
        return 3
    if total / 1e9 > args.max_gb:
        print(f"larger than --max-gb {args.max_gb}: refusing", file=sys.stderr)
        return 3
    if free_gb - total / 1e9 < args.keep_free_gb:
        print(
            f"would leave under {args.keep_free_gb} GB free: refusing", file=sys.stderr
        )
        return 3
    if target.exists() and target.stat().st_size == total:
        print("already complete")
    else:
        download(url, target, total)
    if hash_field:
        name = hash_field[0].lower()
        algorithm = (
            next(a for a in ("sha256", "sha1", "md5") if a in name)
            if any(a in name for a in ("sha256", "sha1", "md5"))
            else None
        )
        if algorithm:
            ok = digest(target, algorithm) == str(hash_field[1]).lower()
            print(f"{algorithm} check: {'OK' if ok else 'MISMATCH'}")
            if not ok:
                return 4
    print(f"done: {target} ({target.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
