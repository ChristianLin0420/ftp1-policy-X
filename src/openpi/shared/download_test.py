import io

from openpi.shared import download


class _FileSystem:
    def __init__(self, payload: bytes):
        self.payload = payload

    def info(self, _url: str) -> dict[str, object]:
        return {"type": "file", "size": len(self.payload), "name": "asset.bin"}

    def open(self, _url: str, _mode: str) -> io.BytesIO:
        return io.BytesIO(self.payload)


def test_download_fsspec_streams_file_to_exact_target(monkeypatch, tmp_path):
    payload = b"ftp1" * 1024
    monkeypatch.setattr(download.fsspec.core, "url_to_fs", lambda *_args, **_kwargs: (_FileSystem(payload), "asset"))

    target = tmp_path / "nested" / "asset.partial"
    download._download_fsspec("gs://bucket/asset.bin", target)  # noqa: SLF001

    assert target.read_bytes() == payload
