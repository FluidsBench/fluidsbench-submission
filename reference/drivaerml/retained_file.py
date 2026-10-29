"""Retained regular-file identities for hash-then-parse evaluator inputs.

Some libraries only accept a filename (not an already-open stream).  This
helper opens the participant or scoring-support pathname exactly once, hashes
that descriptor, and exposes the same inode through ``/proc/self/fd`` or
``/dev/fd``.  Mutation-sensitive metadata can then be checked after every
consumer, including after a memory-mapped array has been reduced.
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


class RetainedFileError(ValueError):
    """Raised when a retained evaluator input cannot be proven immutable."""


@dataclass(frozen=True)
class RetainedFileSnapshot:
    device: int
    inode: int
    mode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_handle(
        cls, handle: BinaryIO, *, label: str
    ) -> "RetainedFileSnapshot":
        try:
            value = os.fstat(handle.fileno())
        except OSError as error:
            raise RetainedFileError(
                f"cannot fstat retained {label} descriptor: {error}"
            ) from error
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            mode=int(value.st_mode),
            size_bytes=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
            ctime_ns=int(value.st_ctime_ns),
        )


@dataclass
class RetainedVerifiedFile:
    """One regular-file descriptor retained from hashing through consumption."""

    source_path: Path
    label: str
    handle: BinaryIO
    snapshot: RetainedFileSnapshot
    _descriptor_path: Path | None = None
    _closed: bool = False

    @classmethod
    def open(cls, path: str | Path, *, label: str) -> "RetainedVerifiedFile":
        # Normalize ``.``/``..`` and make the path absolute without resolving
        # the final component.  Resolving first would silently turn a final
        # symlink into its target before ``O_NOFOLLOW`` could reject it.
        source_path = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
        descriptor: int | None = None
        no_follow = getattr(os, "O_NOFOLLOW", None)
        if no_follow is None:
            raise RetainedFileError(
                "safe retained-file opening requires O_NOFOLLOW support"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow
        try:
            descriptor = os.open(source_path, flags)
            handle = os.fdopen(descriptor, "rb", buffering=0)
            descriptor = None
            snapshot = RetainedFileSnapshot.from_handle(handle, label=label)
            if not stat.S_ISREG(snapshot.mode):
                raise RetainedFileError(
                    f"retained {label} is not a regular file: {source_path}"
                )
            return cls(
                source_path=source_path,
                label=label,
                handle=handle,
                snapshot=snapshot,
            )
        except OSError as error:
            raise RetainedFileError(f"cannot open {label} {source_path}: {error}") from error
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            if "handle" in locals():
                handle.close()
            raise

    def __enter__(self) -> "RetainedVerifiedFile":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - defensive interpreter cleanup.
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        if not self._closed:
            self.handle.close()
            self._closed = True

    def assert_unchanged(self, *, context: str) -> None:
        if self._closed:
            raise RetainedFileError(f"retained {self.label} descriptor is closed")
        current = RetainedFileSnapshot.from_handle(self.handle, label=self.label)
        pathname: os.stat_result | None = None
        try:
            pathname = os.stat(self.source_path, follow_symlinks=False)
            pathname_identity = (int(pathname.st_dev), int(pathname.st_ino))
        except OSError:
            pathname_identity = None
        if (
            current != self.snapshot
            or not (
                pathname is not None
                and stat.S_ISREG(int(pathname.st_mode))
            )
            or pathname_identity != (
                self.snapshot.device,
                self.snapshot.inode,
            )
        ):
            raise RetainedFileError(
                f"retained {self.label} {self.source_path} changed {context}; "
                "the pathname no longer names the retained inode, or device, "
                "inode, mode, size, mtime, or ctime differs from the pre-hash "
                "fstat snapshot"
            )

    def sha256(self, *, chunk_bytes: int = 64 * 1024 * 1024) -> str:
        if (
            not isinstance(chunk_bytes, int)
            or isinstance(chunk_bytes, bool)
            or chunk_bytes < 1
        ):
            raise RetainedFileError("hash chunk size must be a positive integer")
        self.assert_unchanged(context="before hashing")
        digest = hashlib.sha256()
        try:
            self.handle.seek(0)
            while block := self.handle.read(chunk_bytes):
                digest.update(block)
        except OSError as error:
            raise RetainedFileError(
                f"cannot hash retained {self.label} descriptor: {error}"
            ) from error
        self.assert_unchanged(context="while hashing")
        return digest.hexdigest()

    def descriptor_path(self) -> Path:
        self.assert_unchanged(context="before exposing its descriptor path")
        if self._descriptor_path is not None:
            return self._descriptor_path
        descriptor = self.handle.fileno()
        for directory in (Path("/proc/self/fd"), Path("/dev/fd")):
            candidate = directory / str(descriptor)
            try:
                identity = os.stat(candidate)
            except OSError:
                continue
            if (
                int(identity.st_dev) == self.snapshot.device
                and int(identity.st_ino) == self.snapshot.inode
                and stat.S_ISREG(int(identity.st_mode))
            ):
                self._descriptor_path = candidate
                return candidate
        raise RetainedFileError(
            f"no safe descriptor filesystem exposes retained {self.label}; "
            "expected /proc/self/fd or /dev/fd"
        )


__all__ = [
    "RetainedFileError",
    "RetainedFileSnapshot",
    "RetainedVerifiedFile",
]
