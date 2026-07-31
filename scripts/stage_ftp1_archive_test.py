import io
import pathlib
import subprocess
import tarfile

SCRIPT = pathlib.Path(__file__).with_name("stage_ftp1_archive.sh")


def _write_split_tar(parts_dir: pathlib.Path, *, skip_part: int | None = None) -> None:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        payload = b"ftp1\n"
        info = tarfile.TarInfo("sample.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    archive_bytes = stream.getvalue()
    chunk_size = len(archive_bytes) // 3
    chunks = [archive_bytes[:chunk_size], archive_bytes[chunk_size : 2 * chunk_size], archive_bytes[2 * chunk_size :]]
    for index, chunk in enumerate(chunks):
        if index != skip_part:
            (parts_dir / f"sample.tar.part-{index:04d}").write_bytes(chunk)


def test_stage_split_archive(tmp_path):
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    _write_split_tar(parts_dir)
    output = tmp_path / "output"

    subprocess.run(["bash", str(SCRIPT), str(parts_dir), str(output)], check=True)

    assert (output / "sample.txt").read_text() == "ftp1\n"


def test_stage_split_archive_rejects_missing_part(tmp_path):
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    _write_split_tar(parts_dir, skip_part=1)

    result = subprocess.run(
        ["bash", str(SCRIPT), "--check-only", str(parts_dir), str(tmp_path / "output")],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "not consecutive" in result.stderr
