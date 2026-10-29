"""Validate the immutable support metadata; remote payloads are hash-pinned, not downloaded."""
from pathlib import Path
import hashlib, sys
ROOT = Path(__file__).resolve().parents[2]
PINS = {'scripts/validate_native_support_publication.py': 'cbc8cc3ca08cb22311e014a0613e1d950c72ceb8a290681df63568b7559e95dc', 'scripts/validate_scoring_supports.py': '6fece08053c12aa3f5413ce6bad70acac079f560d5aa936692a1c07d566ec4d9', 'reference/scoring_support.py': '8b3cfe26785ba39e5801ef5553700e2fd211cf595c67d5de5b746ffba8ce6948'}
for file, digest in PINS.items():
    if hashlib.sha256((ROOT / file).read_bytes()).hexdigest() != digest:
        raise ValueError(f"publication validator input differs: {file}")
sys.path.insert(0, str(ROOT))
from scripts.validate_native_support_publication import validate
if __name__ == "__main__":
    print(validate('drivaerml', 'drivaerml-native-support-v3'))
