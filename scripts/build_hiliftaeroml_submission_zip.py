#!/usr/bin/env python3
"""Build a deterministic ZIP from an assembled HiLiftAeroML submission.

The source directory is treated as an immutable package tree.  Only regular
files are archived; directory entries are omitted and every member receives
fixed metadata.  This utility packages an already assembled candidate and does
not change benchmark activation, evaluator binding, or submission status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path


RECEIPT_SCHEMA = "hiliftaeroml-deterministic-submission-zip-v1"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
FIXED_MEMBER_MODE = stat.S_IFREG | 0o644
ZIP_COMPRESSION_LEVEL = 9
_COPY_BUFFER_BYTES = 1024 * 1024


class DeterministicZipError(ValueError):
    """Raised when a source tree cannot be packaged safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(_COPY_BUFFER_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _is_within(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def _normalized_member_name(relative_path: Path) -> str:
    name = relative_path.as_posix()
    parts = relative_path.parts
    if (
        relative_path.is_absolute()
        or not parts
        or any(part in {"", ".", ".."} for part in parts)
        or "\\" in name
        or name.startswith("/")
    ):
        raise DeterministicZipError(
            f"source entry does not have a normalized POSIX relative path: "
            f"{relative_path!s}"
        )
    return name


def _inventory_regular_files(
    source_directory: Path,
) -> list[tuple[str, Path, os.stat_result]]:
    files: list[tuple[str, Path, os.stat_result]] = []

    def visit(directory: Path) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = list(iterator)
        except OSError as error:
            raise DeterministicZipError(
                f"cannot scan source directory {directory}: {error}"
            ) from error
        for entry in entries:
            entry_path = Path(entry.path)
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise DeterministicZipError(
                    f"cannot inspect source entry {entry_path}: {error}"
                ) from error
            relative = entry_path.relative_to(source_directory)
            member_name = _normalized_member_name(relative)
            if stat.S_ISLNK(entry_stat.st_mode):
                raise DeterministicZipError(
                    f"symbolic links are not permitted in the source tree: {relative}"
                )
            if stat.S_ISDIR(entry_stat.st_mode):
                visit(entry_path)
            elif stat.S_ISREG(entry_stat.st_mode):
                files.append((member_name, entry_path, entry_stat))
            else:
                raise DeterministicZipError(
                    f"non-regular source entry is not permitted: {relative}"
                )

    visit(source_directory)
    files.sort(key=lambda item: item[0])
    return files


def _zip_member(name: str) -> zipfile.ZipInfo:
    member = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIMESTAMP)
    member.compress_type = zipfile.ZIP_DEFLATED
    member.create_system = 3
    member.external_attr = FIXED_MEMBER_MODE << 16
    member.internal_attr = 0
    member.flag_bits = 0
    member._compresslevel = ZIP_COMPRESSION_LEVEL
    return member


def _copy_regular_file(
    archive: zipfile.ZipFile,
    *,
    member_name: str,
    source_path: Path,
    inventoried_stat: os.stat_result,
) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source_path, flags)
    except OSError as error:
        raise DeterministicZipError(
            f"cannot open regular source file {source_path}: {error}"
        ) from error
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise DeterministicZipError(
                f"source entry is no longer a regular file: {source_path}"
            )
        if (
            opened_stat.st_dev != inventoried_stat.st_dev
            or opened_stat.st_ino != inventoried_stat.st_ino
        ):
            raise DeterministicZipError(
                f"source entry changed during archive construction: {source_path}"
            )
        with os.fdopen(descriptor, "rb", closefd=True) as source:
            descriptor = -1
            with archive.open(
                _zip_member(member_name),
                mode="w",
                force_zip64=True,
            ) as destination:
                shutil.copyfileobj(
                    source,
                    destination,
                    length=_COPY_BUFFER_BYTES,
                )
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def build_deterministic_submission_zip(
    source_directory: Path,
    output_archive: Path,
) -> dict[str, object]:
    """Build one deterministic archive and return its canonical receipt fields."""

    source_directory = Path(source_directory)
    output_archive = Path(output_archive)
    try:
        source_lstat = source_directory.lstat()
    except FileNotFoundError as error:
        raise DeterministicZipError(
            f"source directory does not exist: {source_directory}"
        ) from error
    except OSError as error:
        raise DeterministicZipError(
            f"cannot inspect source directory {source_directory}: {error}"
        ) from error
    if stat.S_ISLNK(source_lstat.st_mode):
        raise DeterministicZipError(
            f"source directory must not be a symbolic link: {source_directory}"
        )
    if not stat.S_ISDIR(source_lstat.st_mode):
        raise DeterministicZipError(
            f"source path is not a directory: {source_directory}"
        )

    source_resolved = source_directory.resolve(strict=True)
    output_resolved = output_archive.resolve(strict=False)
    if _is_within(output_resolved, source_resolved):
        raise DeterministicZipError(
            "output archive must be located outside the source directory"
        )

    if output_archive.is_symlink():
        raise DeterministicZipError(
            f"output archive must not be a symbolic link: {output_archive}"
        )
    try:
        output_lstat = output_archive.lstat()
    except FileNotFoundError:
        output_lstat = None
    except OSError as error:
        raise DeterministicZipError(
            f"cannot inspect output archive {output_archive}: {error}"
        ) from error
    if output_lstat is not None and not stat.S_ISREG(output_lstat.st_mode):
        raise DeterministicZipError(
            f"existing output archive is not a regular file: {output_archive}"
        )

    files = _inventory_regular_files(source_resolved)
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{output_archive.name}.",
        suffix=".tmp",
        dir=output_archive.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            with zipfile.ZipFile(
                handle,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=ZIP_COMPRESSION_LEVEL,
                allowZip64=True,
                strict_timestamps=True,
            ) as archive:
                for member_name, source_path, inventoried_stat in files:
                    _copy_regular_file(
                        archive,
                        member_name=member_name,
                        source_path=source_path,
                        inventoried_stat=inventoried_stat,
                    )
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        if output_archive.is_symlink():
            raise DeterministicZipError(
                f"output archive became a symbolic link: {output_archive}"
            )
        os.replace(temporary, output_archive)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "schema": RECEIPT_SCHEMA,
        "archive": str(output_archive.resolve(strict=True)),
        "archive_sha256": _sha256_file(output_archive),
        "member_count": len(files),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic ZIP from an assembled HiLiftAeroML "
            "submission directory."
        )
    )
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("output_archive", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = build_deterministic_submission_zip(
            args.source_directory,
            args.output_archive,
        )
    except (DeterministicZipError, OSError, zipfile.BadZipFile) as error:
        raise SystemExit(f"deterministic ZIP build failed: {error}") from error
    print(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FIXED_MEMBER_MODE",
    "FIXED_ZIP_TIMESTAMP",
    "RECEIPT_SCHEMA",
    "ZIP_COMPRESSION_LEVEL",
    "DeterministicZipError",
    "build_deterministic_submission_zip",
    "main",
    "parse_args",
]
