"""Tests for rl/oracleArchive.py. Run with: uv run pytest rl/test_oracleArchive.py -q"""
import json

import pyarrow.parquet as pq

from rl import oracleArchive


def cacheFiles(directory):
    """Three cache records across two regions, written as baseline.solveWindow does."""
    directory.mkdir()
    names = {
        "NSW1_2024-01-01_0400_aaaaaaaaaaaaaaaa.json": 1.25,
        "NSW1_2024-01-02_0400_bbbbbbbbbbbbbbbb.json": 0.1 + 0.2,
        "SA1_2024-01-01_0400_cccccccccccccccc.json": -3.5,
    }
    for name, profit in names.items():
        record = {"start": "2024-01-01T04:00:00", "profitPerInterval": [profit, 0.0],
                  "profit": profit, "dualBound": profit, "cyclesUsed": 0.9975,
                  "status": "Optimal", "fallback": None}
        (directory / name).write_text(json.dumps(record))
    return sorted(directory.glob("*.json"))


class TestArchive:
    def test_one_parquet_per_region(self, tmp_path):
        written = oracleArchive.pack(cacheFiles(tmp_path / "json"), tmp_path / "set")
        assert [path.name for path in written] == ["NSW1.parquet", "SA1.parquet"]
        assert pq.read_table(written[0]).num_rows == 2

    def test_unpack_restores_identical_cache_files(self, tmp_path):
        """Same names and same bytes, so solveWindow serves them under the same key."""
        files = cacheFiles(tmp_path / "json")
        oracleArchive.pack(files, tmp_path / "set")
        assert oracleArchive.unpack(tmp_path / "set", tmp_path / "cache") == 3
        assert oracleArchive.lookup(tmp_path / "set", files[2].name) == files[2].read_text()
        for path in files:
            assert (tmp_path / "cache" / path.name).read_bytes() == path.read_bytes()

    def test_unpack_leaves_an_existing_cache_file_alone(self, tmp_path):
        files = cacheFiles(tmp_path / "json")
        oracleArchive.pack(files, tmp_path / "set")
        (tmp_path / "cache").mkdir()
        (tmp_path / "cache" / files[0].name).write_text("local")
        assert oracleArchive.unpack(tmp_path / "set", tmp_path / "cache") == 2
        assert (tmp_path / "cache" / files[0].name).read_text() == "local"

    def test_pack_merges_into_a_set(self, tmp_path):
        """Adding a day keeps the days already packed, and a key packed again replaces
        its old record."""
        first, *rest = cacheFiles(tmp_path / "json")
        oracleArchive.pack(rest, tmp_path / "set")
        first.write_text(first.read_text().replace("1.25", "9.5"))
        oracleArchive.pack([first], tmp_path / "set")
        assert pq.read_table(tmp_path / "set" / "NSW1.parquet").num_rows == 2
        assert "9.5" in oracleArchive.lookup(tmp_path / "set", first.name)

    def test_the_summary_columns_read_without_the_records(self, tmp_path):
        oracleArchive.pack(cacheFiles(tmp_path / "json"), tmp_path / "set")
        table = pq.read_table(tmp_path / "set" / "SA1.parquet",
                              columns=["key", "profit", "status"])
        assert table.column("profit").to_pylist() == [-3.5]
        assert table.column("status").to_pylist() == ["Optimal"]
