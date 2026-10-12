"""Download the Nymeria minimal subset listed in Nymeria_download_urls.json.

Groups taken: body motion, head and wrist recordings, metadata JSON and the two
narration CSVs. Files are fetched to ``<name>.part`` (resumed with an HTTP Range
request when a partial file exists), checked against the SHA-1 in the JSON and
renamed on success. A finished file whose SHA-1 matches is skipped; a mismatch
triggers a fresh download. Standard library only.

JSON layout expected (per sequence, per group)::

    {"sequences": {"<sequence name>": {"<group>": {"download_url": ...,
        "filename": ..., "sha1sum": ..., "file_size_bytes": ...}}}}
"""
import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

NEEDED_GROUPS = ["body_motion", "recording_head", "recording_lwrist",
                 "recording_rwrist", "metadata_json", "narration_atomic_action_csv",
                 "narration_activity_summarization_csv"]
CHUNK = 1 << 20


def sha1_of(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def collect(json_path, only):
    """Return a list of (sequence, group, entry) for the needed groups."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sequences = data.get("sequences", data)
    items = []
    for seq, groups in sequences.items():
        if only and not any(seq.endswith(uid) for uid in only):
            continue
        if not isinstance(groups, dict):
            continue
        for group in NEEDED_GROUPS:
            entry = groups.get(group)
            if entry:
                items.append((seq, group, entry))
    return items


def entry_size(entry):
    return int(entry.get("file_size_bytes") or entry.get("size") or 0)


def entry_sha1(entry):
    return (entry.get("sha1sum") or entry.get("sha1") or "").lower()


def fetch(url, part, expected_size):
    """Download url into part, resuming from its current size."""
    start = part.stat().st_size if part.exists() else 0
    if expected_size and start > expected_size:
        part.unlink()
        start = 0
    if expected_size and start == expected_size:
        return
    req = urllib.request.Request(url)
    if start:
        req.add_header("Range", f"bytes={start}-")
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # range not satisfiable: restart from zero
            part.unlink()
            return fetch(url, part, expected_size)
        raise
    with resp:
        mode = "ab" if start and resp.status == 206 else "wb"
        with open(part, mode) as f:
            while True:
                block = resp.read(CHUNK)
                if not block:
                    break
                f.write(block)


def download(seq, group, entry, out_dir):
    url = entry["download_url"]
    name = entry.get("filename") or Path(url.split("?")[0]).name
    dest_dir = out_dir / seq
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest, part = dest_dir / name, dest_dir / (name + ".part")
    want = entry_sha1(entry)
    if dest.exists():
        if not want or sha1_of(dest) == want:
            print(f"skip (verified)  {seq}/{name}")
            return True
        print(f"sha1 mismatch, re-downloading  {seq}/{name}")
        dest.unlink()
    print(f"download  {seq}/{name}")
    fetch(url, part, entry_size(entry))
    if want and sha1_of(part) != want:
        print(f"FAILED sha1 check  {seq}/{name}", file=sys.stderr)
        part.unlink()
        return False
    part.replace(dest)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", required=True, help="Nymeria_download_urls.json")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--dry-run", action="store_true", help="list files, download nothing")
    ap.add_argument("--list-sizes", action="store_true", help="print sizes per group and total")
    ap.add_argument("--only", nargs="+", default=None,
                    help="space-separated UIDs; keeps sequences whose name ends with one")
    args = ap.parse_args()

    items = collect(args.json, args.only)
    if not items:
        print("no matching files", file=sys.stderr)
        return 1

    if args.list_sizes:
        per_group = {}
        for _, group, entry in items:
            per_group[group] = per_group.get(group, 0) + entry_size(entry)
        for group in NEEDED_GROUPS:
            if group in per_group:
                print(f"{group:42s} {per_group[group] / 1e9:10.3f} GB")
        print(f"{'total':42s} {sum(per_group.values()) / 1e9:10.3f} GB "
              f"({len({s for s, _, _ in items})} sequences, {len(items)} files)")
    if args.dry_run:
        for seq, group, entry in items:
            print(f"{seq}/{entry.get('filename', group)}  {entry_size(entry)} bytes")
        return 0
    if args.list_sizes:
        return 0

    out_dir = Path(args.out)
    failed = sum(not download(s, g, e, out_dir) for s, g, e in items)
    print(f"done: {len(items) - failed} ok, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
