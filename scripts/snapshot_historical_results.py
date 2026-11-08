"""Copy the approved original Git blobs, never the working-tree result tables."""

import argparse
from pathlib import Path

from scmap.historical_snapshot import create_snapshot

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
create_snapshot(Path(__file__).resolve().parents[1], args.output)
