"""The public profile scoring contract is authoritative and rejects altered bindings."""
import copy
import json
from pathlib import Path

import pytest

from reference.drivaerml import dev_profiles as profiles

ROOT = Path(__file__).resolve().parents[1]


def specification():
    return json.loads((ROOT / 'benchmark-specs/drivaerml/submission-spec.json').read_text())


def test_official_profile_contract_is_bound():
    spec = specification()
    profiles.validate_scoring_contract(spec)
    contract = json.loads((ROOT / 'benchmark-specs/drivaerml' / spec['profile_scoring']['file']).read_text())
    assert contract['profile_evidence_schema'] == profiles.SCHEMA
    assert contract['release_id'] == profiles.RELEASE_ID
    assert contract['status'] == 'official'


@pytest.mark.parametrize(('field', 'value'), [
    ('file', 'other-contract.json'),
    ('status', 'dev_only'),
    ('sha256', '0' * 64),
    ('release_id', 'other-release'),
])
def test_wrong_profile_contract_binding_is_rejected(field, value):
    spec = copy.deepcopy(specification())
    spec['profile_scoring'][field] = value
    with pytest.raises(ValueError, match='profile scoring contract differs'):
        profiles.validate_scoring_contract(spec)


def test_changed_scoring_rules_are_rejected(tmp_path, monkeypatch):
    spec = specification()
    path = tmp_path / 'benchmark-specs/drivaerml/profile-scoring-v2.json'
    path.parent.mkdir(parents=True)
    contract = json.loads((ROOT / path.relative_to(tmp_path)).read_text())
    contract['relative_families_weight'] = 1.0
    path.write_text(json.dumps(contract))
    monkeypatch.setattr(profiles, '__file__', str(tmp_path / 'reference/drivaerml/dev_profiles.py'))
    with pytest.raises(ValueError, match='profile scoring contract differs'):
        profiles.validate_scoring_contract(spec)
