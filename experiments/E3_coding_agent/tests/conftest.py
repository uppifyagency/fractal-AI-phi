import sys
from pathlib import Path

# E3's scripts are not an installed package; put them on the path for the tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
