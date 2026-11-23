"""Apply the first-stage structural criteria to generated MR-TADF candidates."""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "results"
DEFAULT_INPUT = (
    RESULTS_DIR / "generated_scaffold_database" / "generated_candidates_valid_unique.csv"
)
DEFAULT_OUTPUT = RESULTS_DIR / "structural_filter" / "scaffold_based_generated_molecules.csv"


def filter_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    # Use the descriptors recorded at generation time and preserve candidate order.
    keep = (
        candidates["is_novel_vs_training"]
        & candidates["MW"].between(350, 1200)
        & candidates["HeavyAtomCount"].between(25, 90)
        & candidates["NumAromaticRings"].between(4, 14)
        & candidates["LogP"].le(15)
        & candidates["RotatableBonds"].le(12)
        & candidates["num_modifications"].le(5)
    )
    return candidates.loc[keep]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    candidates = pd.read_csv(args.input)
    filtered = filter_candidates(candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.output, index=False)
    print(f"Retained {len(filtered):,} of {len(candidates):,} candidates.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
