"""Download immutable documentation archives and assemble a versioned Pages site."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

_VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\Z")
_REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_ASSET_PREFIX = "trajviz-docs-v"
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_MEMBERS = 10_000


def validate_version(value: object) -> str:
    if not isinstance(value, str) or not _VERSION.fullmatch(value):
        raise ValueError("Version must be a plain X.Y.Z release version.")
    return value


def archive_name(version: str) -> str:
    return f"{_ASSET_PREFIX}{validate_version(version)}.zip"


def _gh_json(arguments: list[str]) -> Any:
    result = subprocess.run(["gh", "api", *arguments], capture_output=True, text=True, check=False)
    if result.returncode:
        raise ValueError(f"GitHub API request failed (exit {result.returncode}).")
    try:
        return json.loads(result.stdout)
    except ValueError:
        raise ValueError("GitHub API returned invalid JSON.") from None


def _download_asset(repository: str, asset_id: int, destination: Path) -> None:
    with destination.open("wb") as handle:
        result = subprocess.run(
            ["gh", "api", "-H", "Accept: application/octet-stream",
             f"repos/{repository}/releases/assets/{asset_id}"],
            stdout=handle, stderr=subprocess.PIPE, check=False,
        )
    if result.returncode:
        raise ValueError(f"GitHub archive download failed (exit {result.returncode}).")
    if destination.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Downloaded documentation archive exceeds the size limit.")
    if not zipfile.is_zipfile(destination):
        raise ValueError("Downloaded documentation asset is not a ZIP archive.")


def _release_version(release: dict[str, Any]) -> str:
    tag = release.get("tag_name")
    if not isinstance(tag, str):
        raise ValueError("GitHub release has no version tag.")
    return validate_version(tag[1:] if tag.startswith("v") else tag)


def download_archives(
    repository: str,
    directory: Path,
    metadata: Path,
    *,
    rebuild_stable: bool = False,
    release_version: str | None = None,
    reconcile_from: str | None = None,
) -> dict[str, Any]:
    """Download named release assets through gh; never upload or alter a release."""
    if not isinstance(repository, str) or not _REPOSITORY.fullmatch(repository):
        raise ValueError("Repository must be an owner/name pair.")
    if release_version is not None:
        validate_version(release_version)
    if reconcile_from is not None:
        validate_version(reconcile_from)
    latest = _gh_json([f"repos/{repository}/releases/latest"])
    if not isinstance(latest, dict) or latest.get("draft") or latest.get("prerelease"):
        raise ValueError("The latest release must be a published stable release.")
    latest_version = _release_version(latest)
    pages = _gh_json(["--paginate", "--slurp", f"repos/{repository}/releases"])
    if not isinstance(pages, list) or any(not isinstance(page, list) for page in pages):
        raise ValueError("GitHub release pagination has an unexpected layout.")
    releases = [release for page in pages for release in page]
    if any(not isinstance(release, dict) for release in releases):
        raise ValueError("GitHub release list contains an invalid record.")
    # /latest can briefly lead the paginated listing when a release is created.
    releases = [latest, *releases]
    seen_releases: set[str] = set()
    assets: dict[str, dict[str, Any]] = {}
    for release in releases:
        if release.get("draft") or release.get("prerelease"):
            continue
        try:
            version = _release_version(release)
        except ValueError:
            # Prerelease/custom tags are not part of the stable version archive.
            continue
        # Merge duplicate observations: an asset seen by either endpoint
        # must not be mistaken for a missing archive during API cache races.
        seen_releases.add(version)
        candidates = release.get("assets", [])
        if not isinstance(candidates, list) or any(not isinstance(asset, dict) for asset in candidates):
            raise ValueError("GitHub release assets have an unexpected layout.")
        expected = archive_name(version)
        matches = []
        for asset in candidates:
            name = asset.get("name")
            if isinstance(name, str) and name.startswith(_ASSET_PREFIX) and name.endswith(".zip"):
                if name != expected:
                    raise ValueError("Documentation asset version does not match its release.")
                matches.append(asset)
        if len(matches) > 1:
            raise ValueError("A release has duplicate documentation archive assets.")
        if not matches:
            continue
        asset = matches[0]
        identity, size = asset.get("id"), asset.get("size")
        if type(identity) is not int or identity <= 0 or type(size) is not int or not 0 <= size <= MAX_ARCHIVE_BYTES:
            raise ValueError("Documentation archive asset metadata is invalid or exceeds the size limit.")
        assets[version] = {"version": version, "filename": expected, "asset_id": identity}

    if release_version is not None and release_version not in seen_releases:
        raise ValueError("Requested release must exist and be a published, non-prerelease version.")
    build_versions = []
    if rebuild_stable or latest_version not in assets:
        build_versions.append(latest_version)
    if (release_version is not None and release_version not in assets and release_version not in build_versions
            and (reconcile_from is None or _version_key(release_version) >= _version_key(reconcile_from))):
        build_versions.append(release_version)
    if reconcile_from is not None:
        for candidate in sorted(seen_releases, key=_version_key, reverse=True):
            if (_version_key(candidate) >= _version_key(reconcile_from)
                    and candidate not in assets and candidate not in build_versions):
                build_versions.append(candidate)

    directory.mkdir(parents=True, exist_ok=True)
    for version in sorted(assets, key=_version_key, reverse=True):
        asset = assets[version]
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".docs-download-", suffix=".zip", delete=False) as handle:
                temporary = Path(handle.name)
            _download_asset(repository, asset["asset_id"], temporary)
            os.replace(temporary, directory / asset["filename"])
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    result = {
        "repository": repository,
        "latest_version": latest_version,
        "release_tag": latest["tag_name"],
        "needs_build": bool(build_versions),
        "build_versions": build_versions,
        "archives": [assets[version] for version in sorted(assets, key=_version_key, reverse=True)],
    }
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if output := os.getenv("GITHUB_OUTPUT"):
        with Path(output).open("a", encoding="utf-8") as handle:
            handle.write(f"latest_version={latest_version}\nneeds_build={str(result['needs_build']).lower()}\n")
            handle.write("build_versions=" + json.dumps(build_versions, separators=(",", ":")) + "\n")
    return result


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(component) for component in validate_version(version).split("."))


def _member_path(name: str) -> PurePosixPath:
    if not name or "\\" in name or "\x00" in name or ":" in name:
        raise ValueError("Documentation archive contains an unsafe path.")
    parts = name.rstrip("/").split("/")
    if any(part in {"", ".", ".."} for part in parts) or name.startswith("/"):
        raise ValueError("Documentation archive contains an unsafe path.")
    return PurePosixPath(*parts)


def extract_archive(archive: Path, destination: Path) -> None:
    """Extract validated regular files, without allowing path or symlink escapes."""
    if destination.is_symlink() or (destination.exists() and (not destination.is_dir() or any(destination.iterdir()))):
        raise ValueError("Archive destination must be absent or an empty regular directory.")
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Documentation archive exceeds the size limit.")
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if len(members) > MAX_MEMBERS:
            raise ValueError("Documentation archive contains too many entries.")
        total = 0
        seen: set[str] = set()
        entries: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
        for member in members:
            path = _member_path(member.orig_filename)
            identity = str(path).casefold()
            if identity in seen:
                raise ValueError("Documentation archive contains duplicate paths.")
            seen.add(identity)
            mode = stat.S_IFMT(member.external_attr >> 16)
            if member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise ValueError("Documentation archive uses an unsupported compression method.")
            if mode not in {0, stat.S_IFREG, stat.S_IFDIR} or member.flag_bits & 1:
                raise ValueError("Documentation archives may contain only unencrypted regular files and directories.")
            if mode == stat.S_IFDIR and not member.is_dir():
                raise ValueError("Documentation archive has an invalid directory entry.")
            if member.file_size > MAX_MEMBER_BYTES:
                raise ValueError("Documentation archive member exceeds the size limit.")
            total += member.file_size
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError("Documentation archive exceeds the uncompressed size limit.")
            entries.append((member, path))
        destination.mkdir(parents=True, exist_ok=True)
        for member, relative in entries:
            target = destination.joinpath(*relative.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as incoming, target.open("xb") as outgoing:
                written = 0
                while block := incoming.read(1024 * 1024):
                    written += len(block)
                    if written > MAX_MEMBER_BYTES or written > member.file_size:
                        raise ValueError("Documentation archive member exceeds its declared size.")
                    outgoing.write(block)
            if written != member.file_size:
                raise ValueError("Documentation archive member has an invalid size.")


def _site_files(directory: Path) -> list[Path]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("Documentation source must be a regular directory.")
    files = []
    total = 0
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("Documentation sources may not contain symlinks or special files.")
        if path.is_file():
            size = path.stat().st_size
            if size > MAX_MEMBER_BYTES:
                raise ValueError("Documentation source member exceeds the size limit.")
            total += size
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError("Documentation source exceeds the size limit.")
            _member_path(path.relative_to(directory).as_posix())
            files.append(path)
            if len(files) > MAX_MEMBERS:
                raise ValueError("Documentation source contains too many files.")
    if not (directory / "index.html").is_file():
        raise ValueError("Documentation source has no index.html.")
    return files


def create_archive(source: Path, destination: Path) -> None:
    files = _site_files(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".docs-archive-", suffix=".zip", delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.relative_to(source).as_posix())
        if temporary.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Documentation archive exceeds the size limit.")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def assemble_site(
    dev_site: Path,
    release_site: Path | None,
    version: str,
    archives: Path,
    output: Path,
    archive_output: Path | None = None,
) -> dict[str, Any]:
    """Retain archived HTML unchanged, add the release if built, and create aliases."""
    validate_version(version)
    _site_files(dev_site)
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError("The assembled output directory must be absent or empty.")
    if release_site is not None:
        if archive_output is None:
            raise ValueError("Archive output is required when a newly built release site is provided.")
        _site_files(release_site)
    discovered: dict[str, Path] = {}
    if archives.exists() and not archives.is_dir():
        raise ValueError("The archive location must be a directory.")
    if archives.is_dir():
        for path in archives.glob(f"{_ASSET_PREFIX}*.zip"):
            candidate = validate_version(path.name[len(_ASSET_PREFIX):-4])
            discovered[candidate] = path
    if release_site is None and version not in discovered:
        raise ValueError("The latest release has no documentation archive or newly built site.")
    if release_site is not None:
        create_archive(release_site, archive_output)
        discovered[version] = archive_output
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".docs-site-", dir=output.parent) as temporary:
        stage = Path(temporary) / "site"
        stage.mkdir()
        for candidate, archive in discovered.items():
            destination = stage / candidate
            extract_archive(archive, destination)
            if not (destination / "index.html").is_file():
                raise ValueError("Documentation archive has no index.html.")
        shutil.copytree(stage / version, stage / "stable")
        shutil.copytree(dev_site, stage / "dev")
        versions = {"stable": version, "versions": sorted(discovered, key=_version_key, reverse=True), "dev": "dev"}
        (stage / "versions.json").write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
        (stage / "index.html").write_text(
            '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
            '<meta http-equiv="refresh" content="0;url=stable/">'
            '<link rel="canonical" href="stable/"><title>TrajViz documentation</title>'
            '</head><body><a href="stable/">Open stable documentation</a></body></html>\n',
            encoding="utf-8",
        )
        (stage / ".nojekyll").touch()
        if output.exists():
            output.rmdir()
        stage.rename(output)
    return versions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download existing documentation release assets")
    download.add_argument("--repository", required=True)
    download.add_argument("--directory", type=Path, required=True)
    download.add_argument("--metadata", type=Path, required=True)
    download.add_argument("--rebuild-stable", action="store_true")
    download.add_argument("--release-version", help="Also archive this published maintenance release when missing")
    download.add_argument("--reconcile-from", help="Retry missing published version archives from this initial docs version")
    pack = commands.add_parser("pack", help="Create a validated documentation ZIP without uploading it")
    pack.add_argument("--site", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    assemble = commands.add_parser("assemble", help="Assemble release archives, stable and development sites")
    assemble.add_argument("--dev-site", type=Path, required=True)
    assemble.add_argument("--release-site", type=Path)
    assemble.add_argument("--version", required=True)
    assemble.add_argument("--archives", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.add_argument("--archive-output", type=Path, help="Required only with --release-site")
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            result = download_archives(args.repository, args.directory, args.metadata,
                                       rebuild_stable=args.rebuild_stable, release_version=args.release_version,
                                       reconcile_from=args.reconcile_from)
        elif args.command == "pack":
            create_archive(args.site, args.output)
            result = {"archive": args.output.name, "bytes": args.output.stat().st_size}
        else:
            result = assemble_site(args.dev_site, args.release_site, args.version,
                                   args.archives, args.output, args.archive_output)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        print(f"docs archive: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
