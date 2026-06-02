import os

import pandas as pd

from filtering import apply_report_filters, druglike_score_from_row, weighted_present
from shared_utils import normalize_id, safe_float


def normalize_series(series, higher_is_better=True):
    vals = [x for x in series if x is not None and not pd.isna(x)]
    if not vals:
        return [None] * len(series)
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [0.5 if x is not None and not pd.isna(x) else None for x in series]
    out = []
    for x in series:
        if x is None or pd.isna(x):
            out.append(None)
        else:
            z = (x - lo) / (hi - lo)
            out.append(z if higher_is_better else 1.0 - z)
    return out


def load_external_interaction_counts(csv_path, id_col="Title", interaction_col="interaction_count", *, eprint=None):
    empty_meta = {"ext_rows": 0, "ext_unique_ids": 0, "ext_duplicate_ids": 0, "ext_matched_molecules": 0}
    if not csv_path:
        return None, empty_meta
    if not os.path.exists(csv_path):
        if eprint is not None:
            eprint(f"Warning: external interaction CSV not found: {csv_path}")
        return None, empty_meta

    ext_df = pd.read_csv(csv_path, low_memory=False)
    if id_col not in ext_df.columns or interaction_col not in ext_df.columns:
        raise ValueError(f"External interaction CSV must contain '{id_col}' and '{interaction_col}'")

    hbond_cols = [
        col for col in ext_df.columns
        if isinstance(col, str) and (col.endswith("_donor") or col.endswith("_acceptor"))
    ]
    keep_cols = [id_col, interaction_col, *hbond_cols]
    work = ext_df[keep_cols].copy()
    work["mol_id_norm"] = work[id_col].astype(str).map(normalize_id)
    work["external_interaction_count"] = work[interaction_col].map(safe_float)
    work = work.dropna(subset=["mol_id_norm", "external_interaction_count"])

    for col in hbond_cols:
        work[col] = work[col].fillna("").astype(str)

    duplicate_ids = int((work["mol_id_norm"].value_counts() > 1).sum())
    agg_map = {"external_interaction_count": "max"}
    for col in hbond_cols:
        agg_map[col] = lambda series: next((text for text in series if str(text).strip()), "")
    merged = work.groupby("mol_id_norm", as_index=False).agg(agg_map)
    return merged, {
        "ext_rows": int(len(work)),
        "ext_unique_ids": int(work["mol_id_norm"].nunique()),
        "ext_duplicate_ids": duplicate_ids,
        "ext_matched_molecules": 0,
    }


def merge_and_rank_molecules(
    mol_df,
    ext_df,
    *,
    score_weight,
    interaction_weight,
    max_molecular_weight,
    max_hbd,
    max_hba,
    max_rot_bonds,
    max_formal_charge,
    neutral_only,
):
    out = mol_df.copy()
    out["interaction_count"] = None

    if ext_df is not None:
        out = out.merge(ext_df, on="mol_id_norm", how="left")
    else:
        out["external_interaction_count"] = None

    out["interaction_count"] = out["external_interaction_count"]
    out["interaction_count_source"] = [
        "filtered_csv" if (x is not None and not pd.isna(x)) else "none"
        for x in out["external_interaction_count"]
    ]
    merge_counts = {
        "matched_molecules": int(out["external_interaction_count"].notna().sum()),
    }
    merge_counts["unmatched_molecules"] = int(len(out)) - merge_counts["matched_molecules"]
    merge_counts["ext_matched_molecules"] = merge_counts["matched_molecules"]

    score_norm = normalize_series(out["score"].tolist(), higher_is_better=False)
    interaction_norm = normalize_series(out["interaction_count"].tolist(), higher_is_better=True)

    out["score_norm"] = score_norm
    out["interaction_norm"] = interaction_norm
    out["interaction_term"] = interaction_norm

    sw = max(0.0, float(score_weight))
    iw = max(0.0, float(interaction_weight))
    out["priority_score"] = [
        weighted_present([(score, sw), (interaction_term, iw)])
        for score, interaction_term in zip(out["score_norm"], out["interaction_term"])
    ]
    out["priority_score"] = out["priority_score"].fillna(-1.0)
    out["druglike_score"] = out.apply(druglike_score_from_row, axis=1)
    out["druglike_score"] = out["druglike_score"].fillna(0.0)
    out["presentation_score"] = [
        weighted_present([(priority, 0.55), (druglike, 0.45)])
        for priority, druglike in zip(out["priority_score"], out["druglike_score"])
    ]
    out["presentation_score"] = out["presentation_score"].fillna(-1.0)

    out = apply_report_filters(
        out,
        max_molecular_weight=max_molecular_weight,
        max_hbd=max_hbd,
        max_hba=max_hba,
        max_rot_bonds=max_rot_bonds,
        max_formal_charge=max_formal_charge,
        neutral_only=neutral_only,
    )
    out["presentation_score_final"] = out["presentation_score"]
    out.loc[~out["report_eligible"], "presentation_score_final"] = out.loc[~out["report_eligible"], "presentation_score_final"] - 0.25
    out["presentation_score_final"] = out["presentation_score_final"].fillna(-1.0)
    out["overall_score"] = out["presentation_score_final"].round(3)
    out["priority_rank"] = out["priority_score"].rank(method="dense", ascending=False).astype(int)

    report_mol_df = out[out["report_eligible"]].copy()
    report_mol_df["priority_rank"] = report_mol_df["priority_score"].rank(method="dense", ascending=False).astype(int)
    precluster_hbd_violations = int(
        (pd.to_numeric(report_mol_df["filter_hbd"], errors="coerce") > float(max_hbd)).fillna(False).sum()
        if max_hbd is not None else 0
    )

    return out, report_mol_df, merge_counts, precluster_hbd_violations