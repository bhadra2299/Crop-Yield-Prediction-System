"""Allow python -m agri as an alias for the agri console script."""

from __future__ import annotations

import sys

from agri.cli import main

if __name__ == "__main__":
    sys.exit(main())
