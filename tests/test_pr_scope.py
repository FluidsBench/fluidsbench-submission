"""Exercise the workflow's actual diff command against diverged Git branches."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PullRequestScopeTests(unittest.TestCase):
    def test_upstream_changes_are_not_attributed_to_the_contributor(self):
        workflow = (ROOT / '.github/workflows/validate-submissions.yml').read_text()
        command = next(line.strip() for line in workflow.splitlines()
                       if line.strip().startswith('git diff --no-renames'))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def git(*args):
                return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
            git('init', '-q', '-b', 'main')
            git('config', 'user.name', 'Scope test')
            git('config', 'user.email', 'scope@example.test')
            (root / 'README.md').write_text('Original\n')
            git('add', '.'); git('commit', '-qm', 'Initial source')
            ancestor = git('rev-parse', 'HEAD')
            (root / 'README.md').write_text('Upstream documentation change\n')
            git('commit', '-qam', 'Advance main')
            base = git('rev-parse', 'HEAD')
            git('switch', '-qc', 'participant', ancestor)
            result = 'submissions/airfrans/test-v1/submission.json'
            destination = root / result
            destination.parent.mkdir(parents=True)
            destination.write_text('{}\n')
            git('add', '.'); git('commit', '-qm', 'Submit a result')
            def changed():
                return subprocess.check_output(['bash', '-c', command], cwd=root, text=True,
                    env={**os.environ, 'BASE_SHA': base, 'HEAD_SHA': git('rev-parse', 'HEAD')}).splitlines()
            self.assertEqual(changed(), [result])
            # The common-ancestor diff must still expose forbidden contributor edits.
            (root / 'README.md').write_text('Contributor changed this too\n')
            git('commit', '-qam', 'Forbidden edit')
            self.assertEqual(changed(), ['README.md', result])
