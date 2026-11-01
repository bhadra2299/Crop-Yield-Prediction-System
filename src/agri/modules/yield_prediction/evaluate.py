"""Evaluate yield_prediction. Usage: python -m agri.modules.yield_prediction.evaluate [--version VERSION]"""

from __future__ import annotations

import sys

from agri.common.runner import run_evaluate

if __name__ == "__main__":
    sys.exit(run_evaluate("yield_prediction"))
