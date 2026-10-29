"""Direct entrypoint runner for Open Context Circulator."""

import sys
from pathlib import Path

# Ensure src/ is discoverable when executing main.py directly
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from open_context_circulator.cli import main

if __name__ == "__main__":
    main()
