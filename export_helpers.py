import csv
import os
import re
from concurrent.futures import as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from rdkit import Chem
from rdkit.Chem import Draw

from scaffold_summary_helpers import aligned_depiction_mol, representative_ranked_subset, scaffold_tick_image

try:
    import pyarrow as pa
    import pyarrow.csv as pa_csv
except Exception:
    pa = None
    pa_csv = None


def flush_pending_csv_writes(pending_futures, wait_all=False, max_pending=96):
    if not pending_futures:
        return []
    if wait_all or len(pending_futures) >= max_pending:
        done_iter = as_completed(list(pending_futures)) if wait_all else as_completed(list(pending_futures), timeout=None)
        if wait_all:
            for fut in done_iter:
                fut.result()
                pending_futures.remove(fut)
        else:
            while len(pending_futures) >= max_pending:
                fut = next(done_iter)
                fut.result()
                pending_futures.remove(fut)
    return pending_futures


def write_dataframe_csv(df, out_path, index=False, prefer_arrow=False):
    if prefer_arrow and pa is not None and pa_csv is not None:
        try:
            table = pa.Table.from_pandas(df, preserve_index=bool(index), safe=False)
            write_opts = pa_csv.WriteOptions(include_header=True, delimiter=",", quoting_style="needed")
            pa_csv.write_csv(table, out_path, write_options=write_opts)
            return
        except Exception:
            pass

    try:
        df.to_csv(out_path, index=index, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    except TypeError:
        df.to_csv(out_path, index=index, line_terminator="\n", quoting=csv.QUOTE_MINIMAL)


def make_qc_summary(mol_df, merge_stats):
    qc = []
    qc.append(("n_molecules", len(mol_df)))
    qc.append(("n_unique_mol_ids", int(mol_df["mol_id_norm"].nunique())))
    qc.append(("missing_score_pct", round(100.0 * mol_df["score"].isna().mean(), 2)))
    qc.append(("missing_interaction_count_pct", round(100.0 * mol_df["interaction_count"].isna().mean(), 2)))
    for key, value in merge_stats.items():
        qc.append((key, value))
    return pd.DataFrame(qc, columns=["metric", "value"])


def make_scatter_plot(mol_df, out_png):
    sdf = mol_df.dropna(subset=["score", "interaction_count"]).copy()
    if sdf.empty:
        return
    plt.figure(figsize=(9, 6), dpi=180)
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        try:
            plt.style.use("seaborn-whitegrid")
        except Exception:
            plt.style.use("default")
    sc = plt.scatter(
        sdf["score"],
        sdf["interaction_count"],
        c=sdf.get("priority_score", pd.Series([0.0] * len(sdf))),
        cmap="viridis",
        alpha=0.7,
        s=26,
        edgecolors="none",
    )
    plt.colorbar(sc, label="Priority score")
    plt.xlabel("Docking score (lower is better)")
    plt.ylabel("Interaction count")
    plt.title("Docking Score vs Interaction Count")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def make_central_barplot(central_df, out_png, top_n=20):
    if central_df.empty:
        return
    cdf = central_df.head(top_n).copy().sort_values("central_priority", ascending=True)

    fig, ax = plt.subplots(figsize=(11.5, max(6, 0.75 * len(cdf))), dpi=180)
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        try:
            plt.style.use("seaborn-whitegrid")
        except Exception:
            plt.style.use("default")
    y_pos = list(range(len(cdf)))
    bars = ax.barh(y_pos, cdf["central_priority"], color="#1f77b4", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([""] * len(y_pos))
    xmax = max(float(cdf["central_priority"].max()) * 1.22, 1.0)
    ax.set_xlim(0.0, xmax)
    for ypos, bar, (_, row) in zip(y_pos, bars, cdf.iterrows()):
        x = bar.get_width()
        ax.text(x + 0.01, ypos, f"n={row['n_members']}", va="center", fontsize=8)
        ax.text(xmax * 0.02, ypos, str(row.get("scaffold_name", "")), va="center", ha="left", fontsize=8, color="#1f3551")
        tick_img = scaffold_tick_image(row.get("exact_scaffold_smiles"))
        if tick_img is not None:
            ab = AnnotationBbox(
                OffsetImage(tick_img, zoom=0.42),
                (0.0, ypos),
                xybox=(-58, 0),
                xycoords=("data", "data"),
                boxcoords="offset points",
                box_alignment=(1.0, 0.5),
                frameon=False,
                pad=0.0,
            )
            ab.set_clip_on(False)
            ax.add_artist(ab)
    ax.set_xlabel("Central priority")
    ax.set_ylabel("Scaffold")
    ax.set_title("Top Central Ideas")
    fig.subplots_adjust(left=0.28, right=0.97)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_residue_heatmap(mol_df, top_scaffolds, out_png, max_residues=30):
    contact_cols = [col for col in mol_df.columns if col.endswith("_contact")]
    if not contact_cols or top_scaffolds.empty:
        return

    sub = mol_df[mol_df["scaffold_id"].isin(top_scaffolds["scaffold_id"])].copy()
    if sub.empty:
        return

    sub[contact_cols] = sub[contact_cols].fillna(0.0)
    agg = sub.groupby("scaffold_id")[contact_cols].mean()
    residue_strength = agg.mean(axis=0).sort_values(ascending=False)
    selected_cols = residue_strength.head(max_residues).index.tolist()
    mat = agg[selected_cols]
    residue_labels = [col.rsplit("_", 1)[0] for col in selected_cols]

    plt.figure(figsize=(max(10, 0.3 * len(selected_cols)), max(5, 0.45 * len(mat))), dpi=180)
    plt.imshow(mat.values, aspect="auto", cmap="YlGnBu", interpolation="nearest")
    plt.colorbar(label="Mean residue contact")
    plt.xticks(range(len(residue_labels)), residue_labels, rotation=75, fontsize=8)
    plt.yticks(range(len(mat.index)), [str(x)[:42] for x in mat.index], fontsize=8)
    plt.title("Residue Contact Profile by Top Scaffolds")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def make_central_structure_overview(mol_df, central_df, out_png, top_scaffolds=8, global_reference_smiles=None):
    if central_df.empty:
        return

    chosen = central_df.head(top_scaffolds)
    mols = []
    legends = []
    for _, row in chosen.iterrows():
        sid = row["scaffold_id"]
        sdf = representative_ranked_subset(mol_df[mol_df["scaffold_id"] == sid], top_n=1)
        if sdf.empty:
            continue
        rep = sdf.iloc[0]
        smi = rep.get("smiles")
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol = aligned_depiction_mol(mol, reference=(global_reference_smiles or row.get("exact_scaffold_smiles")))
        mols.append(mol)
        ovr = rep.get("overall_score")
        ovr_str = f"{ovr:.2f}" if ovr is not None and not (isinstance(ovr, float) and pd.isna(ovr)) else "—"
        dl = rep.get("druglike_score")
        dl_str = f"{dl:.2f}" if dl is not None and not (isinstance(dl, float) and pd.isna(dl)) else "—"
        legends.append(
            f"{row.get('scaffold_name')}\n{rep.get('mol_id')}\n"
            f"scr={rep.get('score')} | int={rep.get('interaction_count')} | dl={dl_str} | ovr={ovr_str}"
        )

    if not mols:
        return

    n_cols = min(4, len(mols))
    img = Draw.MolsToGridImage(mols, molsPerRow=n_cols, subImgSize=(340, 280), legends=legends, useSVG=False)
    img.save(out_png)


def _prefix_if_missing(prefix, text):
    base = str(text or "")
    if base.startswith(f"{prefix}_"):
        return base
    return f"{prefix}_{base}"


def prefix_sdf_block_mol_id(block, scaffold_name):
    if not block:
        return block
    prefix = str(scaffold_name or "").strip()
    if not prefix:
        return block

    lines = block.splitlines(keepends=True)
    if not lines:
        return block

    first_line = lines[0].rstrip("\r\n")
    lines[0] = _prefix_if_missing(prefix, first_line) + "\n"

    title_tags = {"_Name", "Name", "NAME", "TITLE", "Title", "s_m_title"}
    for idx in range(len(lines) - 1):
        line = lines[idx].strip()
        if not line.startswith(">"):
            continue
        match = re.search(r"<([^>]+)>", line)
        if not match or match.group(1) not in title_tags:
            continue
        value_idx = idx + 1
        if value_idx < len(lines):
            value_line = lines[value_idx].rstrip("\r\n")
            if value_line and value_line != "$$$$":
                lines[value_idx] = _prefix_if_missing(prefix, value_line) + "\n"

    return "".join(lines)


def read_sdf_blocks_by_index(sdf_path, indices_set, *, eprint=None):
    if not indices_set or not sdf_path:
        return {}
    blocks = {}
    current_idx = 0
    current_lines = []
    try:
        with open(sdf_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                current_lines.append(line)
                if line.strip() == "$$$$":
                    if current_idx in indices_set:
                        blocks[current_idx] = "".join(current_lines)
                    current_lines = []
                    current_idx += 1
    except Exception as exc:
        if eprint is not None:
            eprint(f"Warning: could not read SDF blocks for export: {exc}")
    return blocks


def sanitize_sdf_blocks_for_viewer(blocks):
    """Convert SDF blocks to V2000 format with Kekulized bond orders for 3Dmol.js rendering.

    The input SDF is often V3000 (from RDKit docking pipelines). While 3Dmol.js supports V3000,
    converting to V2000 with explicit Kekulized bond orders (alternating single/double) ensures
    aromatic bonds render as proper double bonds (two parallel cylinders in stick mode).
    Explicit H atoms are preserved (removeHs=False). Falls back to the original block on error.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return blocks

    out = {}
    for idx, block in blocks.items():
        try:
            mol = Chem.MolFromMolBlock(block, removeHs=False, sanitize=True)
            if mol is not None:
                out[idx] = Chem.MolToMolBlock(mol)
            else:
                out[idx] = block
        except Exception:
            out[idx] = block
    return out


def collect_scaffold_export_data(mol_df, central_df_top, sdf_path, top_per_scaffold, *, eprint=None):
    needed_indices = set()
    scaffold_plans = {}
    for _, row in central_df_top.iterrows():
        sid = row["scaffold_id"]
        sname = str(row.get("scaffold_name", str(sid)[:20]))
        members_df = mol_df[mol_df["scaffold_id"] == sid].copy()
        if "priority_rank" in members_df.columns:
            members_df = members_df.sort_values(["priority_rank", "mol_id"], ascending=[True, True])

        all_members = []
        members_by_index = {}
        for _, mrow in members_df.iterrows():
            score = mrow.get("score")
            interactions = mrow.get("interaction_count")
            canonical_smiles = str(
                mrow.get("search_exact_canonical")
                or mrow.get("exact_canonical_smiles")
                or mrow.get("canonical_smiles")
                or mrow.get("smiles")
                or ""
            )
            mol_index = mrow.get("mol_index")
            member_record = {
                "mol_id": str(mrow.get("mol_id", "")),
                "mol_index": None if mol_index is None or (isinstance(mol_index, float) and pd.isna(mol_index)) else int(mol_index),
                "score": None if (score is None or (isinstance(score, float) and pd.isna(score))) else float(score),
                "interaction_count": None if (interactions is None or (isinstance(interactions, float) and pd.isna(interactions))) else float(interactions),
                "smiles": str(mrow.get("smiles") or ""),
                "canonical_smiles": canonical_smiles,
                "substitution_signature": str(mrow.get("substitution_signature") or ""),
            }
            all_members.append(member_record)
            if member_record["mol_index"] is not None:
                members_by_index[member_record["mol_index"]] = member_record

        display_df = representative_ranked_subset(members_df, top_n=top_per_scaffold)
        display_indices = []
        if "mol_index" in display_df.columns:
            display_indices = [int(i) for i in display_df["mol_index"].dropna().tolist()]
        all_member_indices = []
        if "mol_index" in members_df.columns:
            all_member_indices = [int(i) for i in members_df["mol_index"].dropna().tolist()]
        needed_indices.update(display_indices)
        needed_indices.update(all_member_indices)
        scaffold_plans[sname] = {
            "display_mol_indices": display_indices,
            "all_member_indices": all_member_indices,
            "display_members": [members_by_index[idx] for idx in display_indices if idx in members_by_index],
            "all_members": all_members,
        }

    sdf_blocks = read_sdf_blocks_by_index(sdf_path, needed_indices, eprint=eprint)
    export_data = {}
    for sname, plan in scaffold_plans.items():
        sdf_parts = []
        for idx in plan["display_mol_indices"]:
            block = sdf_blocks.get(idx, "")
            if block:
                block = prefix_sdf_block_mol_id(block, sname)
                if not block.rstrip("\n").endswith("$$$$"):
                    block = block.rstrip("\n") + "\n$$$$\n"
                sdf_parts.append(block)
        all_sdf_parts = []
        for idx in plan["all_member_indices"]:
            block = sdf_blocks.get(idx, "")
            if block:
                block = prefix_sdf_block_mol_id(block, sname)
                if not block.rstrip("\n").endswith("$$$$"):
                    block = block.rstrip("\n") + "\n$$$$\n"
                all_sdf_parts.append(block)
        export_data[sname] = {
            "name": sname,
            "display_members": plan["display_members"],
            "all_members": plan["all_members"],
            "sdf_text": "".join(sdf_parts),
            "all_members_sdf_text": "".join(all_sdf_parts),
        }
    return export_data