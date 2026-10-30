"""Versioned documentation retains releases and extracts archives safely."""

from __future__ import annotations

import importlib.util
import json
import stat
import subprocess
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/docs_archive.py"
_SPEC = importlib.util.spec_from_file_location("trajviz_docs_archive", _SCRIPT)
assert _SPEC and _SPEC.loader
archive = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(archive)


def _zip(path: Path, entries: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for name, content in entries.items():
            handle.writestr(name, content)
    return path


def _site(path: Path, text: str) -> Path:
    path.mkdir(parents=True)
    (path / "index.html").write_text(text, encoding="utf-8")
    (path / "api").mkdir()
    (path / "api/reference.html").write_text("Reference " + text, encoding="utf-8")
    return path


def _release(version: str, *, asset: bool = True) -> dict:
    return {"tag_name": "v" + version, "draft": False, "prerelease": False,
            "assets": [{"id": int(version.replace(".", "")) + 1,
                        "name": archive.archive_name(version), "size": 123}] if asset else []}


@pytest.mark.parametrize("version", ["1.2.3", "0.3.1", "10.20.30"])
def test_plain_versions_are_accepted(version):
    assert archive.validate_version(version) == version
    assert archive.archive_name(version) == f"trajviz-docs-v{version}.zip"


@pytest.mark.parametrize("version", ["v0.3.1", "01.2.3", "1.2", "1.2.3-rc1", "1.2.3+local", "../1.2.3", "1.2.3\n", True])
def test_other_version_forms_are_rejected(version):
    with pytest.raises(ValueError, match="plain X.Y.Z"):
        archive.validate_version(version)


def test_download_collects_published_archives_and_exposes_latest_build_need(tmp_path, monkeypatch):
    latest, older = _release("0.3.1", asset=False), _release("0.3.0")
    draft = _release("0.2.0") | {"draft": True}
    custom = {"tag_name": "v0.4.0-rc1", "assets": [], "draft": False}
    downloads = []

    def download(repository, identity, destination):
        downloads.append((repository, identity, destination.name))
        _zip(destination, {"index.html": "older content"})

    output = tmp_path / "outputs.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    with patch.object(archive, "_gh_json", side_effect=[latest, [[latest, older], [draft, custom]]]), patch.object(archive, "_download_asset", side_effect=download):
        metadata = archive.download_archives("rshu/TrajViz", tmp_path / "archives", tmp_path / "metadata.json")
    assert metadata["latest_version"] == "0.3.1"
    assert metadata["release_tag"] == "v0.3.1"
    assert metadata["needs_build"] is True
    assert [entry["version"] for entry in metadata["archives"]] == ["0.3.0"]
    assert len(downloads) == 1 and downloads[0][0] == "rshu/TrajViz"
    assert (tmp_path / "archives/trajviz-docs-v0.3.0.zip").is_file()
    assert json.loads((tmp_path / "metadata.json").read_text()) == metadata
    assert output.read_text() == 'latest_version=0.3.1\nneeds_build=true\nbuild_versions=["0.3.1"]\n'


@pytest.mark.parametrize("rebuild,expected", [(False, False), (True, True)])
def test_download_existing_latest_is_reused_unless_rebuild_requested(tmp_path, monkeypatch, rebuild, expected):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    latest = _release("0.3.1")
    with patch.object(archive, "_gh_json", side_effect=[latest, [[latest]]]), patch.object(archive, "_download_asset") as download:
        download.side_effect = lambda _repository, _identity, path: _zip(path, {"index.html": "release"})
        result = archive.download_archives("rshu/TrajViz", tmp_path / "archives", tmp_path / "metadata.json", rebuild_stable=rebuild)
    assert result["needs_build"] is expected
    assert result["build_versions"] == (["0.3.1"] if rebuild else [])
    assert len(result["archives"]) == 1


@pytest.mark.parametrize("change", [
    {"name": "trajviz-docs-v0.3.0.zip"}, {"id": True}, {"id": -1},
    {"size": -1}, {"size": archive.MAX_ARCHIVE_BYTES + 1},
])
def test_invalid_asset_metadata_is_rejected_before_download(tmp_path, change):
    latest = _release("0.3.1")
    latest["assets"][0].update(change)
    with (
        patch.object(archive, "_gh_json", side_effect=[latest, [[latest]]]),
        patch.object(archive, "_download_asset") as download,
        pytest.raises(ValueError),
    ):
        archive.download_archives("rshu/TrajViz", tmp_path / "archives", tmp_path / "metadata.json")
    download.assert_not_called()


def test_github_errors_do_not_echo_credential_like_stderr():
    result = subprocess.CompletedProcess([], 1, "private-api-key", "private-api-key")
    with patch.object(archive.subprocess, "run", return_value=result), pytest.raises(ValueError) as exc:
        archive._gh_json(["repos/rshu/TrajViz/releases/latest"])
    assert "private-api-key" not in str(exc.value)


@pytest.mark.parametrize("repository", ["https://github.com/rshu/TrajViz", "owner/repo?token=private", "../repo", "owner/repo/extra"])
def test_repository_validation_prevents_arbitrary_api_targets(tmp_path, repository):
    with patch.object(archive, "_gh_json") as request, pytest.raises(ValueError, match="owner/name"):
        archive.download_archives(repository, tmp_path, tmp_path / "meta.json")
    request.assert_not_called()


@pytest.mark.parametrize("name", ["../escape.html", "/escape.html", "a/../../escape", r"a\escape", "C:/escape", "a/./escape"])
def test_unsafe_member_paths_never_escape_destination(tmp_path, name):
    source = _zip(tmp_path / "bad.zip", {name: "unsafe"})
    with pytest.raises(ValueError, match="unsafe path"):
        archive.extract_archive(source, tmp_path / "site")
    assert not (tmp_path / "site").exists()
    assert not (tmp_path / "escape.html").exists()


@pytest.mark.parametrize("mode", [stat.S_IFLNK, stat.S_IFIFO])
def test_symlinks_and_special_files_are_rejected(tmp_path, mode):
    source = tmp_path / "special.zip"
    with zipfile.ZipFile(source, "w") as handle:
        member = zipfile.ZipInfo("link")
        member.create_system = 3
        member.external_attr = (mode | 0o777) << 16
        handle.writestr(member, "../outside")
    with pytest.raises(ValueError, match="regular files"):
        archive.extract_archive(source, tmp_path / "site")
    assert not (tmp_path / "site").exists()


def test_archive_rejects_case_colliding_paths(tmp_path):
    source = _zip(tmp_path / "duplicate.zip", {"Index.html": "one", "index.html": "two"})
    with pytest.raises(ValueError, match="duplicate paths"):
        archive.extract_archive(source, tmp_path / "site")


def test_member_total_and_entry_count_limits_are_enforced(tmp_path):
    source = _zip(tmp_path / "large.zip", {"a.html": "a" * 160, "b.html": "b" * 160})
    with patch.object(archive, "MAX_MEMBER_BYTES", 159), pytest.raises(ValueError, match="member exceeds"):
        archive.extract_archive(source, tmp_path / "member")
    with patch.object(archive, "MAX_ARCHIVE_BYTES", 300), pytest.raises(ValueError, match="uncompressed"):
        archive.extract_archive(source, tmp_path / "total")
    with patch.object(archive, "MAX_MEMBERS", 1), pytest.raises(ValueError, match="too many entries"):
        archive.extract_archive(source, tmp_path / "entries")


def test_existing_destination_cannot_redirect_extraction_through_symlinks(tmp_path):
    source = _zip(tmp_path / "safe.zip", {"index.html": "content"})
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "site"
    destination.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="empty regular directory"):
        archive.extract_archive(source, destination)
    assert not (outside / "index.html").exists()


def test_assembly_retains_old_html_and_creates_version_stable_dev_aliases(tmp_path):
    archives = tmp_path / "archives"
    old = _zip(archives / archive.archive_name("0.3.0"), {"index.html": "immutable old guide", "api.html": "old API"})
    before = old.read_bytes()
    dev = _site(tmp_path / "dev", "development")
    release = _site(tmp_path / "release", "current release")
    output = tmp_path / "published"
    new_archive = tmp_path / "upload.zip"
    result = archive.assemble_site(dev, release, "0.3.1", archives, output, new_archive)
    assert result == {"stable": "0.3.1", "versions": ["0.3.1", "0.3.0"], "dev": "dev"}
    assert old.read_bytes() == before
    assert (output / "0.3.0/index.html").read_text() == "immutable old guide"
    assert (output / "0.3.0/api.html").read_text() == "old API"
    assert (output / "0.3.1/index.html").read_text() == "current release"
    assert (output / "stable/index.html").read_text() == "current release"
    assert (output / "dev/index.html").read_text() == "development"
    assert 'url=stable/' in (output / "index.html").read_text()
    assert json.loads((output / "versions.json").read_text()) == result
    with zipfile.ZipFile(new_archive) as handle:
        assert handle.read("index.html") == b"current release"
    assert (output / ".nojekyll").is_file()


def test_assembly_can_reuse_archives_without_rebuilding_or_writing_new_archive(tmp_path):
    archives = tmp_path / "archives"
    _zip(archives / archive.archive_name("0.10.0"), {"index.html": "latest"})
    _zip(archives / archive.archive_name("0.9.0"), {"index.html": "older"})
    output, unused = tmp_path / "site", tmp_path / "unused.zip"
    result = archive.assemble_site(_site(tmp_path / "dev", "dev"), None, "0.10.0", archives, output, unused)
    assert result["versions"] == ["0.10.0", "0.9.0"]
    assert not unused.exists()
    assert (output / "stable/index.html").read_text() == "latest"


def test_missing_latest_and_nonempty_output_fail_without_modifying_existing_files(tmp_path):
    dev = _site(tmp_path / "dev", "dev")
    with pytest.raises(ValueError, match="no documentation archive"):
        archive.assemble_site(dev, None, "0.3.1", tmp_path / "archives", tmp_path / "site", tmp_path / "upload.zip")
    target = tmp_path / "existing"
    target.mkdir()
    keep = target / "keep.txt"
    keep.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="absent or empty"):
        archive.assemble_site(dev, None, "0.3.1", tmp_path / "archives", target, tmp_path / "upload.zip")
    assert keep.read_text() == "keep"


def test_invalid_archive_does_not_leave_a_partial_published_site(tmp_path):
    archives = tmp_path / "archives"
    _zip(archives / archive.archive_name("0.3.0"), {"../escape": "bad"})
    dev = _site(tmp_path / "dev", "dev")
    release = _site(tmp_path / "release", "release")
    output = tmp_path / "site"
    with pytest.raises(ValueError, match="unsafe path"):
        archive.assemble_site(dev, release, "0.3.1", archives, output, tmp_path / "upload.zip")
    assert not output.exists()
    assert not list(tmp_path.glob(".docs-site-*"))


def test_archive_creation_rejects_symlink_sources(tmp_path):
    source = _site(tmp_path / "source", "source")
    (source / "linked.html").symlink_to(source / "index.html")
    with pytest.raises(ValueError, match="symlinks"):
        archive.create_archive(source, tmp_path / "upload.zip")
    assert not (tmp_path / "upload.zip").exists()


def test_cli_assemble_outputs_machine_readable_versions(tmp_path, capsys):
    dev = _site(tmp_path / "dev", "development")
    release = _site(tmp_path / "release", "release")
    status = archive.main(["assemble", "--dev-site", str(dev), "--release-site", str(release),
                           "--version", "0.3.1", "--archives", str(tmp_path / "archives"),
                           "--output", str(tmp_path / "site"), "--archive-output", str(tmp_path / "upload.zip")])
    assert status == 0
    assert json.loads(capsys.readouterr().out)["stable"] == "0.3.1"


def test_dangling_output_symlink_is_rejected(tmp_path):
    dev = _site(tmp_path / "dev", "dev")
    output = tmp_path / "site"
    output.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(ValueError, match="absent or empty"):
        archive.assemble_site(dev, None, "0.3.1", tmp_path / "archives", output, tmp_path / "upload.zip")
    assert output.is_symlink()


def test_archive_location_must_not_be_a_file(tmp_path):
    dev = _site(tmp_path / "dev", "dev")
    archives = tmp_path / "archives"
    archives.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="archive location"):
        archive.assemble_site(dev, None, "0.3.1", archives, tmp_path / "site", tmp_path / "upload.zip")


def test_nonstandard_compression_is_rejected_before_extraction(tmp_path):
    source = tmp_path / "compression.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_BZIP2) as handle:
        handle.writestr("index.html", "content")
    with pytest.raises(ValueError, match="unsupported compression"):
        archive.extract_archive(source, tmp_path / "site")
    assert not (tmp_path / "site").exists()


def test_plain_version_marked_prerelease_is_not_downloaded(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    latest = _release("0.3.1", asset=False)
    preview = _release("0.4.0") | {"prerelease": True}
    with patch.object(archive, "_gh_json", side_effect=[latest, [[preview, latest]]]), patch.object(archive, "_download_asset") as download:
        metadata = archive.download_archives("rshu/TrajViz", tmp_path / "archives", tmp_path / "meta.json")
    assert metadata["archives"] == []
    assert metadata["build_versions"] == ["0.3.1"]
    download.assert_not_called()


def test_maintenance_event_builds_its_archive_without_changing_latest(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    latest = _release("1.0.0")
    maintenance = _release("0.3.2", asset=False)
    with patch.object(archive, "_gh_json", side_effect=[latest, [[latest, maintenance]]]), patch.object(archive, "_download_asset") as download:
        download.side_effect = lambda _repository, _identity, path: _zip(path, {"index.html": "latest retained"})
        result = archive.download_archives("rshu/TrajViz", tmp_path / "archives", tmp_path / "meta.json", release_version="0.3.2")
    assert result["latest_version"] == "1.0.0"
    assert result["release_tag"] == "v1.0.0"
    assert result["needs_build"] is True
    assert result["build_versions"] == ["0.3.2"]
    assert [item["version"] for item in result["archives"]] == ["1.0.0"]


def test_missing_latest_and_maintenance_event_are_both_built_once(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    latest, maintenance = _release("1.0.0", asset=False), _release("0.3.2", asset=False)
    with patch.object(archive, "_gh_json", side_effect=[latest, [[latest, maintenance]]]):
        result = archive.download_archives("rshu/TrajViz", tmp_path / "archives", tmp_path / "meta.json", release_version="0.3.2")
    assert result["build_versions"] == ["1.0.0", "0.3.2"]
    with patch.object(archive, "_gh_json", side_effect=[latest, [[latest]]]):
        result = archive.download_archives("rshu/TrajViz", tmp_path / "archives", tmp_path / "meta.json", release_version="1.0.0")
    assert result["build_versions"] == ["1.0.0"]


@pytest.mark.parametrize("requested", [_release("0.3.2") | {"prerelease": True},
                                       _release("0.3.2") | {"draft": True}, None])
def test_requested_release_must_exist_and_be_published_stable(tmp_path, requested):
    latest = _release("1.0.0")
    releases = [latest] + ([requested] if requested is not None else [])
    with (patch.object(archive, "_gh_json", side_effect=[latest, [releases]]),
          patch.object(archive, "_download_asset") as download,
          pytest.raises(ValueError, match="published, non-prerelease")):
        archive.download_archives("rshu/TrajViz", tmp_path / "archives", tmp_path / "meta.json", release_version="0.3.2")
    download.assert_not_called()
    assert not (tmp_path / "meta.json").exists()


def test_existing_maintenance_archive_is_not_rebuilt(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    latest, maintenance = _release("1.0.0"), _release("0.3.2")
    with patch.object(archive, "_gh_json", side_effect=[latest, [[latest, maintenance]]]), patch.object(archive, "_download_asset") as download:
        download.side_effect = lambda _repository, _identity, path: _zip(path, {"index.html": "retained"})
        result = archive.download_archives("rshu/TrajViz", tmp_path / "archives", tmp_path / "meta.json", release_version="0.3.2")
    assert result["build_versions"] == [] and result["needs_build"] is False


def test_cli_pack_creates_a_site_archive_without_release_operations(tmp_path, capsys):
    site = _site(tmp_path / "release", "released HTML")
    output = tmp_path / "trajviz-docs-v0.3.1.zip"
    with patch.object(archive, "_gh_json") as request:
        status = archive.main(["pack", "--site", str(site), "--output", str(output)])
    assert status == 0
    assert json.loads(capsys.readouterr().out)["archive"] == output.name
    request.assert_not_called()
    with zipfile.ZipFile(output) as source:
        assert source.read("index.html") == b"released HTML"


def test_maintenance_pack_and_assembly_preserve_stable(tmp_path):
    archives = tmp_path / "archives"
    _zip(archives / archive.archive_name("1.0.0"), {"index.html": "latest"})
    archive.create_archive(_site(tmp_path / "maintenance", "maintenance"), archives / archive.archive_name("0.3.2"))
    result = archive.assemble_site(_site(tmp_path / "dev", "dev"), None, "1.0.0", archives,
                                   tmp_path / "site", tmp_path / "unused.zip")
    assert result["stable"] == "1.0.0"
    assert (tmp_path / "site/stable/index.html").read_text() == "latest"
    assert (tmp_path / "site/0.3.2/index.html").read_text() == "maintenance"


@pytest.mark.parametrize("latest_has_asset", [False, True])
def test_duplicate_release_observations_reuse_asset_seen_by_either_endpoint(tmp_path, monkeypatch, latest_has_asset):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    latest = _release("0.3.1", asset=latest_has_asset)
    page_release = _release("0.3.1", asset=not latest_has_asset)
    with patch.object(archive, "_gh_json", side_effect=[latest, [[page_release]]]), patch.object(archive, "_download_asset") as download:
        download.side_effect = lambda _repository, _identity, path: _zip(path, {"index.html": "immutable"})
        result = archive.download_archives("rshu/TrajViz", tmp_path / "archives", tmp_path / "meta.json")
    assert result["build_versions"] == [] and result["needs_build"] is False
    assert len(result["archives"]) == 1
    download.assert_called_once()


def test_cli_archive_only_assembly_does_not_need_unused_archive_output(tmp_path, capsys):
    dev = _site(tmp_path / "dev", "development")
    archives = tmp_path / "archives"
    _zip(archives / archive.archive_name("0.3.1"), {"index.html": "release"})
    status = archive.main(["assemble", "--dev-site", str(dev), "--version", "0.3.1",
                           "--archives", str(archives), "--output", str(tmp_path / "site")])
    assert status == 0
    assert json.loads(capsys.readouterr().out)["stable"] == "0.3.1"


def test_new_release_assembly_requires_an_explicit_archive_output(tmp_path):
    dev = _site(tmp_path / "dev", "development")
    release = _site(tmp_path / "release", "release")
    with pytest.raises(ValueError, match="Archive output is required"):
        archive.assemble_site(dev, release, "0.3.1", tmp_path / "archives", tmp_path / "site")


def test_packed_archive_also_obeys_compressed_byte_limit(tmp_path):
    site = tmp_path / "source"
    site.mkdir()
    (site / "index.html").write_text("content", encoding="utf-8")
    output = tmp_path / "upload.zip"
    with patch.object(archive, "MAX_ARCHIVE_BYTES", 100), pytest.raises(ValueError, match="size limit"):
        archive.create_archive(site, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".docs-archive-*"))


def test_reconciliation_retries_missing_maintenance_archives_and_prioritizes_event(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    latest = _release("1.0.0")
    releases = [latest, _release("0.3.5", asset=False), _release("0.3.3", asset=False), _release("0.3.2", asset=False)]
    for event, expected in [(None, ["0.3.5", "0.3.3", "0.3.2"]), ("0.3.2", ["0.3.2", "0.3.5", "0.3.3"])]:
        with patch.object(archive, "_gh_json", side_effect=[latest, [releases]]), patch.object(archive, "_download_asset") as download:
            download.side_effect = lambda _repository, _identity, path: _zip(path, {"index.html": "latest retained"})
            result = archive.download_archives("rshu/TrajViz", tmp_path / "archives", tmp_path / "meta.json",
                                               release_version=event, reconcile_from="0.3.1")
        assert result["latest_version"] == "1.0.0"
        assert result["build_versions"] == expected
        assert result["needs_build"] is True


def test_reconciliation_skips_predocs_and_unpublished_versions(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    latest = _release("1.0.0")
    predocs = _release("0.3.0", asset=False)
    releases = [latest, predocs, _release("0.3.3") | {"prerelease": True}, _release("0.3.2") | {"draft": True}]
    with patch.object(archive, "_gh_json", side_effect=[latest, [releases]]), patch.object(archive, "_download_asset") as download:
        download.side_effect = lambda _repository, _identity, path: _zip(path, {"index.html": "latest retained"})
        result = archive.download_archives("rshu/TrajViz", tmp_path / "archives", tmp_path / "meta.json",
                                           release_version="0.3.0", reconcile_from="0.3.1")
    assert result["build_versions"] == []
    assert result["needs_build"] is False
    assert [item["version"] for item in result["archives"]] == ["1.0.0"]
    with patch.object(archive, "_gh_json") as request, pytest.raises(ValueError, match="plain X.Y.Z"):
        archive.download_archives("rshu/TrajViz", tmp_path / "invalid", tmp_path / "invalid.json", reconcile_from="v0.3.1")
    request.assert_not_called()
