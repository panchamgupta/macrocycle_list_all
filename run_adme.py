#!/usr/bin/env python3
"""Run chemprop ADME prediction and back-transform log10 columns to linear scale.

Usage
-----
python run_adme.py \
    --input  direct_linker_enumeration_docking_pose_all_BB_SMILES.csv \
    --output direct_linker_enumeration_docking_pose_all_BB_SMILES_adme.csv \
    [--output-linear direct_linker_enumeration_docking_pose_all_BB_SMILES_adme_linear.csv] \
    [--smiles-column SMILES] \
    [--checkpoint /home/cjamieson/bin/reinvent/adme_models/2026_03_01/model.pt]
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
_CONDA_ENV = "/home/cjamieson/anaconda3/envs/reinvent4.5"
_DEFAULT_CHECKPOINT = "/home/cjamieson/bin/reinvent/adme_models/2026_03_01/model.pt"
_DEFAULT_SMILES_COL = "SMILES"

# Columns predicted as log10 that need back-transforming to linear scale.
LOG10_COLS = [
    "GS_Sol_74",
    "GS_Sol_2",
    "GS_CACO2_A2B_10",
    "GS_CACO2_B2A_10",
    "GS_HP_Free_LT",
    "GS_CACO2_A2B_1",
    "GS_CACO2_B2A_1",
    "GS_HP_Free",
    "GS_Pred_Cl_HLM",
    "GS_MDCK",
    "GS_RED_HP",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _chemprop_bin() -> str:
    candidate = os.path.join(_CONDA_ENV, "bin", "chemprop_predict")
    if os.path.isfile(candidate):
        return candidate
    return "chemprop_predict"


def run_chemprop(test_path: str, preds_path: str, smiles_col: str, checkpoint: str) -> None:
    """Run chemprop_predict using the reinvent4.5 conda environment."""
    python_bin = os.path.join(_CONDA_ENV, "bin", "python")
    if not os.path.isfile(python_bin):
        print(
            f"WARNING: Python not found at {python_bin}. "
            "Attempting to run chemprop_predict on the active PATH.",
            file=sys.stderr,
        )
        python_bin = sys.executable

    chemprop = _chemprop_bin()
    if os.path.isfile(chemprop):
        # Invoke the binary via the env's own interpreter so all env packages are available.
        cmd = [python_bin, chemprop]
    else:
        # Newer chemprop installs expose a module entry point.
        cmd = [python_bin, "-m", "chemprop.train"]

    cmd += [
        "--test_path", test_path,
        "--preds_path", preds_path,
        "--smiles_column", smiles_col,
        "--checkpoint_path", checkpoint,
        "--features_generator", "rdkit_2d_normalized",
        "--no_features_scaling",
    ]

    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        print(f"ERROR: chemprop_predict exited with code {proc.returncode}.", file=sys.stderr)
        sys.exit(proc.returncode)


def back_transform(adme_in: str, adme_out: str) -> None:
    """Back-transform log10-predicted columns to linear scale and write output CSV."""
    adme_df = pd.read_csv(adme_in)

    present = [c for c in LOG10_COLS if c in adme_df.columns]
    missing = [c for c in LOG10_COLS if c not in adme_df.columns]
    if missing:
        print(f"WARNING: columns not found in predictions (skipped): {missing}", file=sys.stderr)

    for col in present:
        adme_df[f"{col}_linear"] = np.power(10.0, adme_df[col].astype(float))

    adme_df.to_csv(adme_out, index=False)

    print(f"\nRows: {len(adme_df):,}")
    print(f"Back-transformed {len(present)} column(s) via 10^x:")
    if present:
        summary = (
            adme_df[[f"{c}_linear" for c in present]]
            .describe(percentiles=[0.25, 0.50, 0.75])
            .T[["min", "25%", "50%", "75%", "max"]]
            .round(4)
        )
        print(summary.to_string())
    print(f"\nWrote: {adme_out}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run chemprop ADME prediction and back-transform log10 columns to linear scale."
    )
    p.add_argument(
        "--input", "-i", required=True, metavar="SMILES_CSV",
        help="Input CSV file with SMILES column (passed to --test_path).",
    )
    p.add_argument(
        "--output", "-o", required=True, metavar="PREDS_CSV",
        help="Output CSV file for raw chemprop predictions (passed to --preds_path).",
    )
    p.add_argument(
        "--output-linear", "-l", metavar="LINEAR_CSV", default=None,
        help=(
            "Output CSV file after log10 back-transformation. "
            "Defaults to <output stem>_linear.csv."
        ),
    )
    p.add_argument(
        "--smiles-column", metavar="COL", default=_DEFAULT_SMILES_COL,
        help=f"Name of the SMILES column in the input CSV (default: {_DEFAULT_SMILES_COL}).",
    )
    p.add_argument(
        "--checkpoint", metavar="PT", default=_DEFAULT_CHECKPOINT,
        help=f"Path to chemprop model checkpoint (default: {_DEFAULT_CHECKPOINT}).",
    )
    p.add_argument(
        "--skip-predict", action="store_true",
        help="Skip chemprop_predict; only run log10 back-transformation on an existing --output file.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.output_linear is None:
        stem, ext = os.path.splitext(args.output)
        args.output_linear = f"{stem}_linear{ext}"

    if not args.skip_predict:
        run_chemprop(
            test_path=args.input,
            preds_path=args.output,
            smiles_col=args.smiles_column,
            checkpoint=args.checkpoint,
        )

    if not os.path.isfile(args.output):
        print(f"ERROR: predictions file not found: {args.output}", file=sys.stderr)
        sys.exit(1)

    back_transform(args.output, args.output_linear)


if __name__ == "__main__":
    main()
