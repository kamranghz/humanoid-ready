"""Read the header of an .mp4 without decoding it and without any dependency.

Prints, per track: handler type, codec, stored width and height, display width and height
(the track header), the rotation implied by the track matrix, frame count, duration and fps.
It cannot show what the pictures look like; extract one frame for that.

Usage:
    python scripts/mp4_probe.py FILE.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


def boxes(data: bytes, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack(">I4s", data[pos : pos + 8])
        header = 8
        if size == 1:
            size = struct.unpack(">Q", data[pos + 8 : pos + 16])[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            return
        yield kind, pos + header, pos + size
        pos += size


def top_level_moov(path: Path) -> bytes:
    """Find the moov box by walking top-level headers with seeks; read only that box."""
    with open(path, "rb") as handle:
        handle.seek(0, 2)
        total = handle.tell()
        pos = 0
        while pos + 8 <= total:
            handle.seek(pos)
            size, kind = struct.unpack(">I4s", handle.read(8))
            header = 8
            if size == 1:
                size = struct.unpack(">Q", handle.read(8))[0]
                header = 16
            elif size == 0:
                size = total - pos
            if size < header:
                break
            if kind == b"moov":
                handle.seek(pos + header)
                return handle.read(size - header)
            pos += size
    raise ValueError("no moov box found")


def parse_trak(data: bytes, start: int, end: int) -> dict:
    info: dict = {}
    stack = [(start, end)]
    while stack:
        lo, hi = stack.pop()
        for kind, body, stop in boxes(data, lo, hi):
            if kind in CONTAINERS:
                stack.append((body, stop))
            elif kind == b"tkhd":
                version = data[body]
                off = 52 if version == 1 else 40
                matrix = struct.unpack(">9i", data[body + off : body + off + 36])
                w, h = struct.unpack(">II", data[body + off + 36 : body + off + 44])
                a, b = matrix[0] / 65536, matrix[1] / 65536
                info["display_wh"] = [w / 65536, h / 65536]
                info["rotation_deg"] = round(math.degrees(math.atan2(b, a))) % 360
            elif kind == b"mdhd":
                version = data[body]
                if version == 1:
                    scale, dur = struct.unpack(">IQ", data[body + 20 : body + 32])
                else:
                    scale, dur = struct.unpack(">II", data[body + 12 : body + 20])
                info["timescale"], info["duration_units"] = scale, dur
            elif kind == b"hdlr":
                info["handler"] = data[body + 8 : body + 12].decode("latin1")
            elif kind == b"stsd":
                entry = body + 8
                _, fourcc = struct.unpack(">I4s", data[entry : entry + 8])
                info["codec"] = fourcc.decode("latin1")
                if info.get("handler") == "vide":
                    w, h = struct.unpack(">HH", data[entry + 32 : entry + 36])
                    info["stored_wh"] = [w, h]
            elif kind == b"stts":
                count = struct.unpack(">I", data[body + 4 : body + 8])[0]
                total = 0
                for i in range(count):
                    n, _ = struct.unpack(
                        ">II", data[body + 8 + 8 * i : body + 16 + 8 * i]
                    )
                    total += n
                info["frames_stts"] = total
            elif kind == b"stsz":
                info["frames_stsz"] = struct.unpack(">I", data[body + 8 : body + 12])[0]
    if info.get("timescale") and info.get("duration_units"):
        info["duration_s"] = round(info["duration_units"] / info["timescale"], 2)
        frames = info.get("frames_stsz") or info.get("frames_stts")
        if frames:
            info["fps"] = round(
                frames / (info["duration_units"] / info["timescale"]), 3
            )
    return info


def probe(path: Path) -> dict:
    moov = top_level_moov(path)
    tracks = [
        parse_trak(moov, body, stop)
        for kind, body, stop in boxes(moov, 0, len(moov))
        if kind == b"trak"
    ]
    return {"file": str(path), "bytes": path.stat().st_size, "tracks": tracks}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("file", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(probe(args.file), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
