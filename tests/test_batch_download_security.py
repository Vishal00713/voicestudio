"""Unit tests verifying path containment and security in batch download endpoint."""
from unittest.mock import MagicMock
import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse


@pytest.fixture(autouse=True)
def mock_torch_if_missing(monkeypatch):
    import sys
    if "torch" not in sys.modules:
        mock_t = MagicMock()
        mock_t.cuda.is_available.return_value = False
        monkeypatch.setitem(sys.modules, "torch", mock_t)


@pytest.fixture
def batch_env(tmp_path, monkeypatch):
    from api.routers import batch

    batch_root = tmp_path / "batch"
    batch_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(batch, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(batch, "_jobs", {})
    return batch, tmp_path


def test_download_batch_output_serves_valid_contained_file(batch_env):
    batch, tmp_path = batch_env
    job_id = "job-safe-123"
    job_dir = tmp_path / "batch" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_file = job_dir / "output_es.mp4"
    output_file.write_bytes(b"dummy mp4 content")

    batch._jobs[job_id] = {
        "status": "done",
        "filename": "my_video.mp4",
        "outputs": {"es": str(output_file)},
    }

    response = batch.download_batch_output(job_id, "es")
    assert isinstance(response, FileResponse)
    assert response.path == str(output_file)
    assert response.media_type == "video/mp4"
    assert "my_video_es.mp4" in response.filename


def test_download_batch_output_rejects_escaping_absolute_path(batch_env):
    batch, tmp_path = batch_env
    job_id = "job-escape-abs"

    # Outside batch root
    outside_file = tmp_path / "secret.txt"
    outside_file.write_bytes(b"sensitive data")

    batch._jobs[job_id] = {
        "status": "done",
        "filename": "my_video.mp4",
        "outputs": {"es": str(outside_file)},
    }

    with pytest.raises(HTTPException) as exc_info:
        batch.download_batch_output(job_id, "es")
    assert exc_info.value.status_code == 400
    assert "Invalid batch output file path" in exc_info.value.detail


def test_download_batch_output_rejects_traversal_relative_path(batch_env):
    batch, tmp_path = batch_env
    job_id = "job-escape-rel"

    batch._jobs[job_id] = {
        "status": "done",
        "filename": "my_video.mp4",
        "outputs": {"es": "../../etc/passwd"},
    }

    with pytest.raises(HTTPException) as exc_info:
        batch.download_batch_output(job_id, "es")
    assert exc_info.value.status_code == 400
    assert "Invalid batch output file path" in exc_info.value.detail


def test_download_batch_output_returns_404_for_missing_file(batch_env):
    batch, tmp_path = batch_env
    job_id = "job-missing-file"
    job_dir = tmp_path / "batch" / job_id
    missing_file = job_dir / "output_es.mp4"

    batch._jobs[job_id] = {
        "status": "done",
        "filename": "my_video.mp4",
        "outputs": {"es": str(missing_file)},
    }

    with pytest.raises(HTTPException) as exc_info:
        batch.download_batch_output(job_id, "es")
    assert exc_info.value.status_code == 404
    assert "No output for language" in exc_info.value.detail


def test_download_batch_output_validates_job_state(batch_env):
    batch, _ = batch_env

    # Job not found
    with pytest.raises(HTTPException) as exc_info:
        batch.download_batch_output("nonexistent-job", "es")
    assert exc_info.value.status_code == 404

    # Job not done
    batch._jobs["job-pending"] = {
        "status": "processing",
        "filename": "video.mp4",
        "outputs": {},
    }
    with pytest.raises(HTTPException) as exc_info:
        batch.download_batch_output("job-pending", "es")
    assert exc_info.value.status_code == 400
    assert "not done" in exc_info.value.detail

    # Language not in outputs
    batch._jobs["job-done-no-lang"] = {
        "status": "done",
        "filename": "video.mp4",
        "outputs": {"fr": "/some/path"},
    }
    with pytest.raises(HTTPException) as exc_info:
        batch.download_batch_output("job-done-no-lang", "es")
    assert exc_info.value.status_code == 404


def test_download_batch_output_supports_symlinked_data_dir(tmp_path, monkeypatch, symlink_or_skip):
    import os
    from api.routers import batch

    real_data = tmp_path / "real_data"
    real_data.mkdir(parents=True, exist_ok=True)
    sym_data = tmp_path / "sym_data"
    symlink_or_skip(str(sym_data), str(real_data), target_is_directory=True)


    monkeypatch.setattr(batch, "DATA_DIR", str(sym_data))
    monkeypatch.setattr(batch, "_jobs", {})

    job_id = "job-symlink-test"
    job_dir = sym_data / "batch" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    out_file = job_dir / "output_es.mp4"
    out_file.write_bytes(b"symlink mp4 data")

    batch._jobs[job_id] = {
        "status": "done",
        "filename": "sym_video.mp4",
        "outputs": {"es": str(out_file)},
    }

    res = batch.download_batch_output(job_id, "es")
    assert isinstance(res, FileResponse)
    assert os.path.realpath(res.path) == os.path.realpath(str(out_file))


def test_download_batch_output_supports_relative_data_dir(tmp_path, monkeypatch):
    import os
    from api.routers import batch

    cwd = os.getcwd()
    rel_data = os.path.relpath(str(tmp_path), cwd)

    monkeypatch.setattr(batch, "DATA_DIR", rel_data)
    monkeypatch.setattr(batch, "_jobs", {})

    job_id = "job-rel-test"
    job_dir = tmp_path / "batch" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    out_file = job_dir / "output_es.mp4"
    out_file.write_bytes(b"rel mp4 data")

    stored_rel_path = os.path.join(rel_data, "batch", job_id, "output_es.mp4")

    batch._jobs[job_id] = {
        "status": "done",
        "filename": "rel_video.mp4",
        "outputs": {"es": stored_rel_path},
    }

    res = batch.download_batch_output(job_id, "es")
    assert isinstance(res, FileResponse)
    assert os.path.realpath(res.path) == os.path.realpath(str(out_file))

