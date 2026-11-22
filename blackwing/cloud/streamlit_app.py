"""Cloud-compatible entrypoint; canonical FastAPI persistence, no session DB."""
import sys
from pathlib import Path

root = str(Path(__file__).resolve().parents[2])
if root not in sys.path:
    sys.path.insert(0, root)

from blackwing.ui import main

if __name__ == "__main__":
    main(cloud=True)
