"""Train yield_prediction. Usage: python -m agri.modules.yield_prediction.train"""

from __future__ import annotations

import sys

from agri.common.runner import run_train

if __name__ == "__main__":
    sys.exit(run_train("yield_prediction"))
