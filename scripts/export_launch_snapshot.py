#!/usr/bin/env python3
"""Export the committed launch tree without Git history or trial submissions.

This command creates local files only. It never creates a remote repository,
changes a branch, publishes releases, or deploys the website.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export(destination: Path) -> None:
    destination = destination.resolve()
    if destination.exists():
        raise ValueError('destination must not already exist')
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    if dirty.strip():
        raise ValueError('commit the reviewed source tree before exporting it')
    # Export HEAD, never a working-directory glob that can copy caches or secrets.
    with tempfile.TemporaryDirectory(prefix='fluidsbench-export-') as temp:
        archive = Path(temp) / 'source.tar'
        subprocess.run(['git', 'archive', '--format=tar', '-o', str(archive), 'HEAD'], cwd=ROOT, check=True)
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            if any(m.name == '.git' or m.name.startswith(('.git/', 'submissions/')) for m in members):
                raise ValueError('launch source still contains history or submission packages')
            destination.mkdir(parents=True)
            tar.extractall(destination, filter='data')
    all_rows = json.loads((destination / 'leaderboard/all.json').read_text())
    claims = json.loads((destination / 'leaderboard/claims/index.json').read_text())
    revisions = json.loads((destination / 'leaderboard/revisions.json').read_text())
    if all_rows or claims['record_count'] or revisions['record_count']:
        raise ValueError('launch feed is not empty')
    print(f'Exported history-free, submission-free launch source to {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    export(parser.parse_args().output)
