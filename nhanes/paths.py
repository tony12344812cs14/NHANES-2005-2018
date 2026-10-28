"""Where data and outputs live. Override with environment variables:

  NHANES_DATA  downloaded NHANES files, one folder per cycle (default: <repo>/data)
  NHANES_WORK  everything the pipeline writes                 (default: <repo>/outputs)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NHANES_DATA", ROOT / "data"))
WORK = Path(os.environ.get("NHANES_WORK", ROOT / "outputs"))
RESULTS = WORK / "modeling"
DRUG_FILE = DATA / "RXQ_DRUG.xpt"  # Multum drug categories, one file for all cycles
THREADS = int(os.environ.get("THREADS", "4"))

WORK.mkdir(parents=True, exist_ok=True)
