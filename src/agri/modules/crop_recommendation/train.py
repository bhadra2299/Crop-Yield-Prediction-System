"""Train crop_recommendation. Usage: python -m agri.modules.crop_recommendation.train"""

from __future__ import annotations

import sys

from agri.common.runner import run_train

if __name__ == "__main__":
    sys.exit(run_train("crop_recommendation"))
