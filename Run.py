"""Run the ColabTensor app.

Usage:
    python Run.py [--session NAME] [--gpu G4] [--only gpt-oss|decider|both]
                  [--dry-run] [--stop] [--health-timeout SECONDS]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "Src"))

from ColabTensor.Main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
