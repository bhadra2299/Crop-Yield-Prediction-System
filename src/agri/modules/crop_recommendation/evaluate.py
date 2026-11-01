"""Evaluate crop_recommendation. Usage: python -m agri.modules.crop_recommendation.evaluate [--version VERSION]"""

from __future__ import annotations

import sys

from agri.common.runner import run_evaluate

if __name__ == "__main__":
    sys.exit(run_evaluate("crop_recommendation"))
