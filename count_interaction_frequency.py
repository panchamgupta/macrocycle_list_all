#!/usr/bin/env python3
import argparse
import pandas as pd
import re

DIGIT_RE = re.compile(r"\d+")
RANGE_RE = re.compile(r"^([A-Za-z]+)?(\d+)\s*-\s*([A-Za-z]+)?(\d+)$")

def cell_has_digits(x) -> bool:
    """True if cell contains at least one digit (atom index), else False."""
    if pd.isna(x):
        return False
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return False
    return bool(DIGIT_RE.search(s))

def expand_token(token: str):
    """
    Expand a residue token that may be a single residue (A157)
    or a range (A150-A160) or (A150-160).
    Returns a list of residues.
    """
    token = token.strip()
    if not token:
        return []

    m = RANGE_RE.match(token.replace(" ", ""))
    if not m:
        # Not a range
        return [token]

    chain1, start, chain2, end = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))

    # Handle "A150-160" where second chain is missing
    if chain1 and not chain2:
        chain2 = chain1

    # If both chains exist and differ, we won't expand (ambiguous); keep as-is
    if chain1 and chain2 and chain1 != chain2:
        return [token]  # or raise, but better to be permissive

    chain = chain1 or chain2 or ""  # allow numeric-only, though your columns use A###
    lo, hi = sorted([start, end])
    return [f"{chain}{i}" for i in range(lo, hi + 1)]

def parse_residues(items):
    """
    Accept residues as:
      --residues A157 A247
      --residues A157,A247
      --residues A150-A160
      --residues A150-160
    and combinations thereof.
    """
    residues = []
    for it in items:
        for part in it.split(","):
            residues.extend(expand_token(part))

    # de-dupe preserving order
    seen = set()
    uniq = []
    for r in residues:
        r = r.strip()
        if r and r not in seen:
            uniq.append(r)
            seen.add(r)
    return uniq

def ensure_columns_exist(df, cols):
    """If a requested donor/acceptor column doesn't exist, create it with NA."""
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def compute_interacts(df, residues, sep=","):
    """
    Creates:
      - interacts: comma-separated list of residues that interact (donor/acceptor only)
      - interaction_count: number of residues that interact
    """
    donor_acceptor_cols = []
    for r in residues:
        donor_acceptor_cols.append(f"{r}_donor")
        donor_acceptor_cols.append(f"{r}_acceptor")

    df = ensure_columns_exist(df, donor_acceptor_cols)

    interacts_vals = []
    counts = []

    for _, row in df.iterrows():
        hits = []
        for r in residues:
            d = f"{r}_donor"
            a = f"{r}_acceptor"
            if cell_has_digits(row[d]) or cell_has_digits(row[a]):
                hits.append(r)
        interacts_vals.append(sep.join(hits))
        counts.append(len(hits))

    df["interacts"] = interacts_vals
    df["interaction_count"] = counts
    return df, donor_acceptor_cols

def main():
    ap = argparse.ArgumentParser(
        description="Output minimal interaction summary: Title, interacts list, interaction_count, and donor/acceptor columns for specified residues."
    )
    ap.add_argument("-i", "--input", required=True, help="Input interaction fingerprint CSV.")
    ap.add_argument("-o", "--output", default="interaction_minimal.csv",
                    help="Output CSV (minimal columns).")
    ap.add_argument("--title-col", default="Title",
                    help="Name of the Title/ID column (default: Title).")
    ap.add_argument("--residues", nargs="+", required=True,
                    help="Residues or ranges, e.g. --residues A157 A247 or --residues A150-A160 or mixed.")
    ap.add_argument("--sep", default=",",
                    help="Separator inside 'interacts' column (default: ',').")
    ap.add_argument("--only-hits", action="store_true",
                    help="If set, keep only rows where interaction_count > 0.")
    ap.add_argument("--require-all", action="store_true",
                    help="If set, keep only rows where interaction_count equals number of residues requested.")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    residues = parse_residues(args.residues)

    if args.title_col not in df.columns:
        raise ValueError(f"Title column '{args.title_col}' not found in input CSV.")

    df, da_cols = compute_interacts(df, residues, sep=args.sep)

    # Optional filters
    if args.only_hits:
        df = df[df["interaction_count"] > 0]
    if args.require_all:
        df = df[df["interaction_count"] == len(residues)]

    # Keep ONLY requested output columns
    out_cols = [args.title_col, "interacts", "interaction_count"] + da_cols
    df_out = df.loc[:, out_cols]

    df_out.to_csv(args.output, index=False)

    # Console summary
    print("\n=== Done ===")
    print(f"Residues requested ({len(residues)}): {residues[:10]}{'...' if len(residues) > 10 else ''}")
    print(f"Rows written: {len(df_out)}")
    print(f"Output file: {args.output}")

if __name__ == "__main__":
    main()