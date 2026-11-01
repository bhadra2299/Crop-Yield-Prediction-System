"""Train fertilizer. Usage: python -m agri.modules.fertilizer.train"""

from __future__ import annotations

import sys

from agri.common.runner import run_train

if __name__ == "__main__":
    sys.exit(run_train("fertilizer"))
