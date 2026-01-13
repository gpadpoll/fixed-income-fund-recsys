import io
import zipfile
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from fif_recsys import commands
from fif_recsys.commands import data


def _make_zip_bytes(filename: str, csv_content: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as z:
        z.writestr(filename, csv_content)
    return buf.getvalue()


class FakeResp:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


def test_download_zip_returns_zip(monkeypatch):
    # Prepare a zip with a single CSV
    content = _make_zip_bytes("data.csv", "col1;col2\n1;2\n")

    def fake_get(url, timeout=60):
        return FakeResp(content)

    monkeypatch.setattr(data, "requests", type("R", (), {"get": staticmethod(fake_get)}))

    z = data.download_zip("http://example.com/test.zip")
    assert isinstance(z, zipfile.ZipFile)
    assert any(name.lower().endswith(".csv") for name in z.namelist())


def test_load_zip_parses_csv_and_sets_period():
    content = _make_zip_bytes("data.csv", "col1;col2\n1;2\n")
    z = zipfile.ZipFile(io.BytesIO(content))

    df = data.load_zip(z, "202501")
    assert isinstance(df, pd.DataFrame)
    assert "period" in df.columns
    assert list(df["period"]) == ["202501"]


def test_fetch_writes_parquet_on_success(monkeypatch, tmp_path):
    # Create manifest with one dataset and one period
    manifest = {
        "myds": {
            "base_url": "http://example.com/",
            "periods": ["202501"],
            "filename_template": "file_{period}.zip",
        }
    }

    # Prepare zip content returned by download_zip
    content = _make_zip_bytes("data.csv", "col1;col2\n1;2\n")

    def fake_download_zip(url):
        return zipfile.ZipFile(io.BytesIO(content))

    monkeypatch.setattr(data, "download_zip", fake_download_zip)

    called = {}

    def fake_to_parquet(self, path, index=False):
        called["path"] = Path(path)
        called["df"] = self

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    # Run fetch programmatically using the helper with a fixed reference_date
    ref = "2026-01-12"
    result = data.fetch_manifest(manifest, tmp_path, reference_date=ref)

    # Ensure returned dict contains the dataset and its dataframe
    assert "myds" in result
    df_returned = result["myds"]

    # Ensure parquet was "written" to expected partition path
    expected = tmp_path / "myds" / "period=202501" / "data.parquet"
    assert called.get("path") == expected

    # Ensure reference_date column was added and set correctly in both returned and written dfs
    df_written = called.get("df")
    assert "reference_date" in df_written.columns
    assert all(df_written["reference_date"] == ref)

    assert "reference_date" in df_returned.columns
    assert all(df_returned["reference_date"] == ref)


def test_fetch_skips_when_all_periods_fail(monkeypatch, tmp_path):
    manifest = {
        "myds": {
            "base_url": "http://example.com/",
            "periods": ["202501"],
            "filename_template": "file_{period}.zip",
        }
    }

    # Make download_zip raise an exception
    def fake_download_zip(url):
        raise RuntimeError("network error")

    monkeypatch.setattr(data, "download_zip", fake_download_zip)

    # Ensure no exception is raised; output files should not be created
    result = data.fetch_manifest(manifest, tmp_path)

    assert not (tmp_path / "myds").exists()
    assert result == {}
