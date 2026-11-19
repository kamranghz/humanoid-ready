"""Read-only inventory of the Nymeria download.

Reports what is on disk without extracting anything: which files each sequence has, the
licence and script/location metadata, the zip member names (central directory only), and
how often the narrations describe floor work by the wearer ("C" in the narration text).

Usage (hready environment, Windows):
    python scripts/nymeria_inventory.py
    python scripts/nymeria_inventory.py --peek 8 --peek-bytes 1500 --npz-keys

Nothing is written inside the repository by default. The full JSON goes next to the data
(``--out``), and only aggregate numbers are printed, so a summary can be copied into docs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

DEFAULT_ROOT = Path(r"D:\projects\hready_data\datasets\nymeria")
DEFAULT_OUT = Path(r"D:\projects\hready_data\review\nymeria_inventory.json")

# Suffix after "Nymeria_v<version>_<sequence>_" for every file a full sequence should have.
EXPECTED = (
    "body_motion.zip",
    "recording_head.zip",
    "recording_lwrist.zip",
    "recording_rwrist.zip",
    "metadata.json",
    "narration_activity_summarization.csv",
    "narration_atomic_action.csv",
)
ZIP_KINDS = (
    "body_motion.zip",
    "recording_head.zip",
    "recording_lwrist.zip",
    "recording_rwrist.zip",
)
TEXT_EXT = {".json", ".jsonl", ".csv", ".txt", ".yaml", ".yml", ".md"}
BODY_NPZ_MEMBER = "body/xdata.npz"

RAISED_SUPPORT = r"\b(bed|sofa|couch|chair|armchair|bench|stool|mattress)\b"
GROUND = r"\b(floor|ground|carpet|rug|mat)\b"
PATTERNS = {
    # Each pattern requires the wearer ("C") earlier in the same sentence.
    "lie": r"\bC\b[^.]{0,80}\b(lies|lie down|lying|lays down|laying down|lay down)\b",
    "kneel": r"\bC\b[^.]{0,80}\b(kneels?|kneeling|knelt)\b",
    "crawl": r"\bC\b[^.]{0,80}\b(crawls?|crawling)\b",
    "squat_or_crouch": r"\bC\b[^.]{0,80}\b(squats?|squatting|crouch(es|ing)?)\b",
    "sit_on_ground": r"\bC\b[^.]{0,80}\b(sits?|sitting)\b[^.]{0,40}" + GROUND,
    "push_up_or_plank": r"\bC\b[^.]{0,80}\b(push.?ups?|plank)\b",
}


def sequence_dirs(root: Path) -> list:
    return sorted(p for p in root.iterdir() if p.is_dir())


def find_file(seq_dir: Path, suffix: str):
    hits = sorted(seq_dir.glob("*_" + suffix))
    return hits[0] if hits else None


def read_metadata(path):
    if path is None:
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def zip_summary(path: Path, member_limit: int = 40) -> dict:
    """Central directory only: no member is read or extracted."""
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
    ext = Counter(
        Path(i.filename).suffix.lower() or "(none)" for i in infos if not i.is_dir()
    )
    return {
        "n_members": len(infos),
        "uncompressed_bytes": int(sum(i.file_size for i in infos)),
        "extensions": dict(ext),
        "first_members": [i.filename for i in infos[:member_limit]],
    }


def peek_zip(path: Path, limit: int, nbytes: int) -> list:
    """Read the head of text members (in memory, nothing is written to disk)."""
    out = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if len(out) >= limit:
                break
            if info.is_dir() or Path(info.filename).suffix.lower() not in TEXT_EXT:
                continue
            with archive.open(info) as handle:
                head = handle.read(nbytes).decode("utf-8", errors="replace")
            out.append({"member": info.filename, "bytes": info.file_size, "head": head})
    return out


def npz_array_headers(path: Path, member: str = BODY_NPZ_MEMBER) -> list:
    """Name, shape and dtype of every array in an .npz stored inside a zip.

    Only the .npy headers are read; no array is loaded and nothing is written to disk.
    """
    import numpy as np

    out = []
    with (
        zipfile.ZipFile(path) as outer,
        outer.open(member) as raw,
        zipfile.ZipFile(raw) as inner,
    ):
        for info in inner.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with inner.open(info) as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
                else:
                    shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
            out.append(
                {
                    "array": info.filename[: -len(".npy")],
                    "shape": list(shape),
                    "dtype": str(dtype),
                }
            )
    return out


def jsonl_structure(path: Path, suffix: str, max_list: int = 6) -> dict:
    """Structure of the first line of the first ``*suffix`` member of a zip.

    Long lists are cut to ``max_list`` items and strings to 80 characters, so the output is small.
    Only that one line is read; nothing is written to disk.
    """
    with zipfile.ZipFile(path) as archive:
        names = [i.filename for i in archive.infolist() if i.filename.endswith(suffix)]
        if not names:
            raise KeyError(f"no member ending in {suffix}")
        with archive.open(names[0]) as handle:
            line = handle.readline()
    return {"member": names[0], "first_line": _shorten(json.loads(line), max_list)}


def _shorten(value, max_list: int):
    if isinstance(value, dict):
        return {key: _shorten(item, max_list) for key, item in value.items()}
    if isinstance(value, list):
        head = [_shorten(item, max_list) for item in value[:max_list]]
        return head + [f"... ({len(value)} items)"] if len(value) > max_list else head
    if isinstance(value, str):
        return value[:80]
    return value


def narration_counts(path) -> dict:
    """Count atomic-action sentences that describe floor-related behaviour of the wearer."""
    counts = Counter()
    n_rows = 0
    if path is None:
        return {"n_rows": 0}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            text = list(row.values())[-1] or ""
            n_rows += 1
            for name, pattern in PATTERNS.items():
                match = re.search(pattern, text)
                if not match:
                    continue
                counts[name] += 1
                if name == "lie":
                    sentence = text[match.start() :]
                    sentence = sentence.split(".")[0]
                    if re.search(RAISED_SUPPORT, sentence, flags=re.IGNORECASE):
                        counts["lie_on_raised_support"] += 1
                    elif re.search(GROUND, sentence, flags=re.IGNORECASE):
                        counts["lie_on_ground"] += 1
                    else:
                        counts["lie_surface_unstated"] += 1
    result = {"n_rows": n_rows}
    result.update(counts)
    return result


def inventory(
    root: Path,
    zip_listing: str,
    peek: int,
    peek_bytes: int,
    npz_keys: bool,
    calibration: bool = False,
    max_list: int = 6,
) -> dict:
    sequences = []
    missing = Counter()
    for seq_dir in sequence_dirs(root):
        files = {s: find_file(seq_dir, s) for s in EXPECTED}
        for suffix, path in files.items():
            if path is None:
                missing[suffix] += 1
        meta = read_metadata(files["metadata.json"])
        entry = {
            "sequence": seq_dir.name,
            "present": sorted(s for s, p in files.items() if p is not None),
            "sizes_bytes": {
                s: p.stat().st_size for s, p in files.items() if p is not None
            },
            "metadata": None
            if meta is None
            else {
                k: meta.get(k)
                for k in (
                    "license",
                    "version",
                    "script",
                    "location",
                    "action_duration_sec",
                    "has_two_participants",
                    "body_motion",
                    "head_data",
                    "head_slam",
                    "atomic_action",
                    "activity_summarization",
                    "participant_height_cm",
                    "participant_age_group",
                )
            },
            "narration_atomic_action": narration_counts(
                files["narration_atomic_action.csv"]
            ),
            "zips": {},
        }
        want_zips = zip_listing == "all" or (
            zip_listing == "first" and len(sequences) < 2
        )
        if want_zips:
            for kind in ZIP_KINDS:
                path = files[kind]
                if path is None:
                    continue
                try:
                    entry["zips"][kind] = zip_summary(path)
                    if peek and len(sequences) < 2:
                        entry["zips"][kind]["peek"] = peek_zip(path, peek, peek_bytes)
                except zipfile.BadZipFile as error:
                    entry["zips"][kind] = {"error": f"BadZipFile: {error}"}
        head_zip = files["recording_head.zip"]
        if calibration and head_zip is not None and not sequences:
            try:
                entry["head_online_calibration"] = jsonl_structure(
                    head_zip, "online_calibration.jsonl", max_list
                )
            except (zipfile.BadZipFile, KeyError, ValueError) as error:
                entry["head_online_calibration"] = {
                    "error": f"{type(error).__name__}: {error}"
                }
        body_zip = files["body_motion.zip"]
        if npz_keys and body_zip is not None and len(sequences) < 2:
            try:
                entry["body_npz_arrays"] = npz_array_headers(body_zip)
            except (zipfile.BadZipFile, KeyError, ValueError) as error:
                entry["body_npz_arrays"] = {"error": f"{type(error).__name__}: {error}"}
        sequences.append(entry)
    return {
        "root": str(root),
        "n_sequences": len(sequences),
        "missing_by_file": dict(missing),
        "sequences": sequences,
    }


def summarize(inv: dict) -> dict:
    seqs = inv["sequences"]
    metas = [s["metadata"] for s in seqs if s["metadata"]]
    hours = sum((m.get("action_duration_sec") or 0.0) for m in metas) / 3600.0
    totals = Counter()
    by_script = {}
    for s in seqs:
        counts = s["narration_atomic_action"]
        for key, value in counts.items():
            totals[key] += value
        script = (s["metadata"] or {}).get("script", "unknown")
        bucket = by_script.setdefault(script, Counter())
        bucket["sequences"] += 1
        for key, value in counts.items():
            bucket[key] += value
    full = sum(1 for s in seqs if len(s["present"]) == len(EXPECTED))
    return {
        "n_sequences": inv["n_sequences"],
        "n_with_all_expected_files": full,
        "missing_by_file": inv["missing_by_file"],
        "licences": dict(Counter((m or {}).get("license") for m in metas)),
        "body_head_slam_all_true": sum(
            1
            for m in metas
            if m.get("body_motion") and m.get("head_data") and m.get("head_slam")
        ),
        "has_two_participants": sum(1 for m in metas if m.get("has_two_participants")),
        "action_hours": round(hours, 2),
        "locations": len({m.get("location") for m in metas}),
        "scripts": dict(Counter(m.get("script") for m in metas)),
        "narration_totals": dict(totals),
        "narration_by_script": {k: dict(v) for k, v in sorted(by_script.items())},
        "total_zip_gb": round(
            sum(
                sz
                for s in seqs
                for k, sz in s["sizes_bytes"].items()
                if k.endswith(".zip")
            )
            / 1e9,
            1,
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--zip-listing",
        choices=("none", "first", "all"),
        default="first",
        help="list zip members for no sequence, the first two, or all (slow)",
    )
    parser.add_argument(
        "--peek",
        type=int,
        default=0,
        metavar="N",
        help="print the head of up to N text members per zip of the first two sequences",
    )
    parser.add_argument(
        "--peek-bytes",
        type=int,
        default=600,
        help="bytes printed from the head of each peeked member",
    )
    parser.add_argument(
        "--npz-keys",
        action="store_true",
        help="print array names, shapes and dtypes of body/xdata.npz for the first two sequences",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="print the structure of the first line of the first sequence's online_calibration.jsonl",
    )
    parser.add_argument(
        "--max-list",
        type=int,
        default=6,
        help="items kept per list when printing the calibration line (use 40 to see all)",
    )
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"root not found: {args.root}", file=sys.stderr)
        return 2
    inv = inventory(
        args.root,
        args.zip_listing,
        args.peek,
        args.peek_bytes,
        args.npz_keys,
        args.calibration,
        args.max_list,
    )
    summary = summarize(inv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, **inv}, handle, indent=2)
    print(json.dumps(summary, indent=2))
    for seq in inv["sequences"][:2]:
        for kind, info in seq["zips"].items():
            print(
                f"\n[{seq['sequence']}] {kind}: {info.get('n_members')} members, "
                f"{info.get('uncompressed_bytes')} bytes, extensions {info.get('extensions')}"
            )
            for member in info.get("first_members", [])[:12]:
                print("   ", member)
            for item in info.get("peek", []):
                print(
                    f"   --- {item['member']} ({item['bytes']} bytes)\n{item['head']}"
                )
    for seq in inv["sequences"][:2]:
        arrays = seq.get("body_npz_arrays")
        if arrays is None:
            continue
        print(f"\n[{seq['sequence']}] {BODY_NPZ_MEMBER} arrays:")
        if isinstance(arrays, dict):
            print("   ", arrays)
            continue
        for item in arrays:
            print(f"    {item['array']}: shape {item['shape']} dtype {item['dtype']}")
    calibration = (
        inv["sequences"][0].get("head_online_calibration") if inv["sequences"] else None
    )
    if calibration is not None:
        print("\nrecording_head online_calibration.jsonl, first line (shortened):")
        print(json.dumps(calibration, indent=1))
    print(f"\nfull inventory written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
