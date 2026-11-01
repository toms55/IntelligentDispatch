"""The tracked oracle sets: one Parquet per region, not one JSON per day.

`rl/baseline.py` caches each oracle window as its own JSON under data_cache/lp_baseline/,
named by `cacheKey`. That cache is per machine and gitignored. `oracleResults/<set>/` is
what git tracks so another machine skips the solves, and one file per day made it
thousands of small files per set, which git and a network filesystem both handle badly.

So a set holds one Parquet per region:

    oracleResults/<set>/<REGION>.parquet   one row per cached window

Each row keeps the cache file's name and its JSON text unchanged. `baseline.findRecord`
serves a record straight out of it under the same key, and `unpack` writes back
byte-identical cache files. A few scalars sit beside the text, so a set can be read
without parsing every record.

    uv run python -m rl.oracleArchive pack <folder of cache JSONs> oracleResults/dp
    uv run python -m rl.oracleArchive unpack oracleResults/dp     # into the local cache
"""
import argparse
import functools
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Scalars copied out of each record for reading a set at a glance. The JSON text stays
# the source of truth.
summaryFields = ("profit", "dualBound", "cyclesUsed", "status")


def region(key):
    """The region a cache key names: <REGION>_<date>_<time>_<digest>.json."""
    return key.split("_", 1)[0]


def _read(path):
    table = pq.read_table(path, columns=["key", "record"])
    return dict(zip(table.column("key").to_pylist(), table.column("record").to_pylist()))


@functools.lru_cache(maxsize=32)
def _regionRecords(path, mtime):
    """One region's {key: JSON text}, read once per file version: a run looks up a
    thousand days, and a pack rewrites the file under a new mtime."""
    return _read(path)


def regionRecords(directory, regionName):
    """{key: JSON text} for one region of a set, or {} when the set has no such file."""
    path = Path(directory) / f"{regionName}.parquet"
    if not path.exists():
        return {}
    return _regionRecords(str(path), path.stat().st_mtime_ns)


def lookup(directory, key):
    """The JSON text a set holds under `key`, or None."""
    return regionRecords(directory, region(key)).get(key)


def pack(files, outDir):
    """Merge cache JSONs into outDir/<REGION>.parquet. Returns the paths written.

    A record already in the set is kept unless a file of the same key replaces it.
    """
    byRegion = {}
    for path in sorted(Path(path) for path in files):
        byRegion.setdefault(region(path.name), []).append(path)
    outDir = Path(outDir)
    outDir.mkdir(parents=True, exist_ok=True)
    written = []
    for regionName, paths in sorted(byRegion.items()):
        target = outDir / f"{regionName}.parquet"
        merged = _read(target) if target.exists() else {}
        merged.update({path.name: path.read_text() for path in paths})
        keys = sorted(merged)
        parsed = [json.loads(merged[key]) for key in keys]
        columns = {
            "key": keys,
            "start": [record.get("start") for record in parsed],
            **{field: [record.get(field) for record in parsed] for field in summaryFields},
            "record": [merged[key] for key in keys],
        }
        table = pa.table({name: pa.array(values) for name, values in columns.items()})
        temporary = target.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary, compression="zstd", compression_level=9)
        temporary.replace(target)
        written.append(target)
    return written


def unpack(archiveDir, cacheDir, overwrite=False):
    """Write every record of a set back as a cache JSON. Returns the count written. A
    cache file already there is left alone unless `overwrite`."""
    cacheDir = Path(cacheDir)
    cacheDir.mkdir(parents=True, exist_ok=True)
    count = 0
    for archive in sorted(Path(archiveDir).glob("*.parquet")):
        for key, text in _read(archive).items():
            path = cacheDir / key
            if path.exists() and not overwrite:
                continue
            path.write_text(text)
            count += 1
    return count


def main(argv=None):
    from rl.baseline import cacheRoot

    parser = argparse.ArgumentParser(prog="python -m rl.oracleArchive",
                                     description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    packer = sub.add_parser("pack", help="merge cache JSONs into a set")
    packer.add_argument("source", help="a folder of cache JSONs")
    packer.add_argument("out", help="the set, e.g. oracleResults/dp")
    unpacker = sub.add_parser("unpack", help="a set back into the cache")
    unpacker.add_argument("archive", help="the set, e.g. oracleResults/dp")
    unpacker.add_argument("--cache-dir", default=str(cacheRoot))
    unpacker.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "pack":
        for path in pack(sorted(Path(args.source).glob("*.json")), args.out):
            print(path)
    else:
        print(f"{unpack(args.archive, args.cache_dir, args.overwrite)} records written "
              f"to {args.cache_dir}")


if __name__ == "__main__":
    main()
