"""Minimal stdlib reader for polygon shapefiles and their dBASE tables.

Enough for HydroBASINS: shape type 5 (polygon), numeric and character fields.
"""

from __future__ import annotations

import struct


def read_dbf(data: bytes):
    n_rec, hdr_len, rec_len = struct.unpack("<IHH", data[4:12])
    fields = []
    pos = 32
    while data[pos] != 0x0D:
        name = data[pos:pos + 11].split(b"\0")[0].decode("ascii")
        ftype = chr(data[pos + 11])
        size = data[pos + 16]
        fields.append((name, ftype, size))
        pos += 32
    rows = []
    for i in range(n_rec):
        rec = data[hdr_len + i * rec_len: hdr_len + (i + 1) * rec_len]
        off = 1
        row = {}
        for name, ftype, size in fields:
            raw = rec[off:off + size].decode("latin-1").strip()
            off += size
            if ftype in "NF":
                try:
                    row[name] = float(raw) if "." in raw else int(raw)
                except ValueError:
                    row[name] = None
            else:
                row[name] = raw
        rows.append(row)
    return rows


def read_shp(data: bytes):
    """Yield (bbox, parts) per record, parts as lists of (lon, lat)."""
    pos = 100
    while pos < len(data):
        _, length = struct.unpack(">II", data[pos:pos + 8])
        body = data[pos + 8: pos + 8 + 2 * length]
        pos += 8 + 2 * length
        stype = struct.unpack("<i", body[:4])[0]
        if stype == 0:
            yield None, []
            continue
        bbox = struct.unpack("<4d", body[4:36])
        n_parts, n_points = struct.unpack("<ii", body[36:44])
        starts = list(struct.unpack(f"<{n_parts}i", body[44:44 + 4 * n_parts]))
        p0 = 44 + 4 * n_parts
        pts = struct.unpack(f"<{2 * n_points}d", body[p0:p0 + 16 * n_points])
        xy = list(zip(pts[0::2], pts[1::2]))
        starts.append(n_points)
        yield bbox, [xy[starts[k]:starts[k + 1]] for k in range(n_parts)]
