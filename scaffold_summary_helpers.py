import base64
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from cli_config import prefixed_output_name
from filtering import weighted_present
from ranking_helpers import normalize_series
from shared_utils import hash_text


def variable_positions_from_members(sdf_members):
    pos_to_values = defaultdict(set)
    for sub_map in sdf_members["substitution_map"].dropna():
        if not isinstance(sub_map, dict):
            continue
        for pos, frags in sub_map.items():
            key = tuple(sorted(frags)) if isinstance(frags, (list, tuple)) else (str(frags),)
            pos_to_values[int(pos)].add(key)
    return sorted(pos for pos, vals in pos_to_values.items() if len(vals) > 1)


def variable_substitution_summary(sdf_members, var_positions, top_n=3):
    pos_counter = defaultdict(Counter)
    for sub_map in sdf_members["substitution_map"].dropna():
        if not isinstance(sub_map, dict):
            continue
        for pos in var_positions:
            if pos in sub_map:
                frags = sub_map[pos]
                key = ".".join(sorted(frags)) if isinstance(frags, (list, tuple)) else str(frags)
                pos_counter[pos][key] += 1
    parts = []
    for pos in sorted(var_positions):
        top = pos_counter[pos].most_common(top_n)
        subs = " / ".join([f"{key} (n={value})" for key, value in top])
        parts.append(f"R{pos + 1}: {subs}")
    return " | ".join(parts) if parts else "all positions constant"


def filter_sig_to_positions(sig_str, var_positions):
    if not sig_str or not var_positions:
        return sig_str or ""
    var_set = {pos + 1 for pos in var_positions}
    filtered = []
    for part in sig_str.split("|"):
        part = part.strip()
        match = re.match(r"R(\d+)=", part)
        if match and int(match.group(1)) in var_set:
            filtered.append(part)
    return " | ".join(filtered) if filtered else "—"


def _scaffold_fp(smiles, radius=2, n_bits=2048):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:
        return None


def tanimoto_similarity(smiles_a, smiles_b):
    fp_a = _scaffold_fp(smiles_a)
    fp_b = _scaffold_fp(smiles_b)
    if fp_a is None or fp_b is None:
        return None
    try:
        return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
    except Exception:
        return None


def select_depiction_reference_smiles(scaf_df, preferred_scaffold_name="SCF-001"):
    if scaf_df is None or scaf_df.empty:
        return None

    pick = scaf_df[scaf_df["scaffold_name"] == preferred_scaffold_name]
    if pick.empty:
        pick = scaf_df.head(1)

    row = pick.iloc[0]
    return row.get("exact_scaffold_smiles") or row.get("representative_smiles") or None


def strip_nonpolar_explicit_hydrogens(mol):
    if mol is None:
        return None
    try:
        rw = Chem.RWMol(Chem.Mol(mol))
        to_remove = []
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() != 1 or atom.GetFormalCharge() != 0 or atom.GetDegree() != 1:
                continue
            nbr = atom.GetNeighbors()[0]
            if nbr.GetAtomicNum() == 6:
                to_remove.append(atom.GetIdx())
        for idx in sorted(to_remove, reverse=True):
            rw.RemoveAtom(idx)
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        return out
    except Exception:
        return Chem.Mol(mol)


def depiction_mol(mol):
    return strip_nonpolar_explicit_hydrogens(mol)


def _coerce_template_mol(reference):
    if isinstance(reference, str) and reference:
        return Chem.MolFromSmiles(reference)
    if reference is None:
        return None
    return Chem.Mol(reference)


def generate_macrocycle_depiction_coords(mol, template_mol=None, *, strict_template=False):
    """Generate macrocycle-friendly 2D coordinates with optional template matching.

    Workflow A: when template_mol is provided, match depiction to template using
    rdDepictor.GenerateDepictionMatching2DStructure.
    Workflow B: without template, generate direct 2D coordinates via Compute2DCoords.
    """
    query = depiction_mol(mol)
    if query is None:
        raise ValueError("failed to prepare molecule for depiction")

    rdDepictor.SetPreferCoordGen(True)
    if template_mol is None:
        rdDepictor.Compute2DCoords(query)
        return query

    ref = depiction_mol(template_mol)
    if ref is None:
        if strict_template:
            raise ValueError("template preparation failed")
        rdDepictor.Compute2DCoords(query)
        return query

    rdDepictor.Compute2DCoords(ref)
    if not query.HasSubstructMatch(ref):
        if strict_template:
            raise ValueError("template mismatch: query does not match reference template")
        rdDepictor.Compute2DCoords(query)
        return query

    try:
        rdDepictor.GenerateDepictionMatching2DStructure(query, ref)
        return query
    except Exception as exc:
        if strict_template:
            raise RuntimeError(f"depiction generation with template failed: {exc}")
        rdDepictor.Compute2DCoords(query)
        return query


def aligned_depiction_mol(mol, reference=None):
    mc = depiction_mol(mol)
    if mc is None:
        return None
    ref_mol = _coerce_template_mol(reference)
    try:
        return generate_macrocycle_depiction_coords(mc, template_mol=ref_mol, strict_template=False)
    except Exception:
        try:
            rdDepictor.Compute2DCoords(mc)
            return mc
        except Exception:
            return None


def write_macrocycle_depiction_png(
    mol,
    out_path,
    *,
    template_mol=None,
    legend="",
    size=(1050, 750),
    strict_template=False,
):
    """Write a macrocycle-friendly PNG depiction to disk.

    Returns
    -------
    tuple(bool, str)
        (success, message). Message is empty on success.
    """
    if mol is None:
        return False, "invalid molecule"
    try:
        dep = generate_macrocycle_depiction_coords(
            mol,
            template_mol=template_mol,
            strict_template=strict_template,
        )
    except Exception as exc:
        return False, str(exc)

    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    except Exception:
        pass

    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(int(size[0]), int(size[1]))
        dopts = drawer.drawOptions()
        dopts.addStereoAnnotation = False
        dopts.explicitMethyl = False
        dopts.fixedBondLength = 24.0
        drawer.DrawMolecule(dep, legend=str(legend or ""))
        drawer.FinishDrawing()
        with open(out_path, "wb") as fh:
            fh.write(drawer.GetDrawingText())
        return True, ""
    except Exception as exc:
        return False, f"failed to render/write depiction PNG: {exc}"


def mol_png_base64(mol, size=(260, 180), legend="", reference=None):
    mc = aligned_depiction_mol(mol, reference=reference)
    if mc is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    dopts = drawer.drawOptions()
    dopts.addStereoAnnotation = False
    dopts.explicitMethyl = False
    # Keep a consistent medicinal-chem style bond length across canvases.
    dopts.fixedBondLength = 24.0
    drawer.DrawMolecule(mc, legend=legend)
    drawer.FinishDrawing()
    data = drawer.GetDrawingText()
    return base64.b64encode(data).decode("ascii")


def mol_png_base64_from_smiles(smiles, size=(260, 180), legend="", reference_smiles=None):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol_png_base64(mol, size=size, legend=legend, reference=reference_smiles)


def scaffold_tick_image(smiles, size=(220, 120)):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    dep = aligned_depiction_mol(mol)
    if dep is None:
        return None
    try:
        return np.asarray(Draw.MolToImage(dep, size=size))
    except Exception:
        return None


def representative_ranked_subset(df, top_n=6):
    pres_col = "presentation_score_final" if "presentation_score_final" in df.columns else "presentation_score"
    return df.sort_values(
        [pres_col, "priority_score", "interaction_count", "score"],
        ascending=[False, False, False, True],
    ).head(top_n)


def annotate_similarity_and_rank_central(scaf_df, min_group_size):
    out = scaf_df.copy()
    out["tanimoto_similarity_to_top"] = np.nan
    out["tanimoto_distance_to_top"] = np.nan
    out["central_rerank_score"] = np.nan
    out["high_distance_central"] = False

    central_df = out[out["n_members"] >= min_group_size].copy()
    if central_df.empty:
        central_df["rank_central"] = pd.Series(dtype=int)
        return out, central_df, None

    top_ref = central_df.sort_values(
        ["n_members", "central_priority", "median_interaction_count"],
        ascending=[False, False, False],
    ).iloc[0]
    reference_smiles = top_ref.get("representative_smiles") or top_ref.get("exact_scaffold_smiles") or None

    sims = []
    dists = []
    for _, row in out.iterrows():
        probe_smiles = row.get("representative_smiles") or row.get("exact_scaffold_smiles")
        sim = tanimoto_similarity(probe_smiles, reference_smiles)
        sims.append(sim)
        dists.append((1.0 - sim) if sim is not None else None)

    out["tanimoto_similarity_to_top"] = sims
    out["tanimoto_distance_to_top"] = dists
    out["central_rerank_score"] = [
        weighted_present([(cp, 0.55), (sim, 0.45)])
        for cp, sim in zip(out["central_priority"], out["tanimoto_similarity_to_top"])
    ]
    out["central_rerank_score"] = out["central_rerank_score"].fillna(out["central_priority"])

    central_df = out[out["n_members"] >= min_group_size].copy()
    dist_vals = pd.to_numeric(central_df["tanimoto_distance_to_top"], errors="coerce").dropna()
    if not dist_vals.empty:
        q70 = float(dist_vals.quantile(0.70))
        central_df["high_distance_central"] = pd.to_numeric(
            central_df["tanimoto_distance_to_top"], errors="coerce"
        ) >= q70
    else:
        central_df["high_distance_central"] = False

    high_distance_map = dict(zip(central_df["scaffold_id"], central_df["high_distance_central"]))
    out["high_distance_central"] = out["scaffold_id"].map(high_distance_map).fillna(False)

    central_df = central_df.sort_values(
        [
            "central_rerank_score",
            "tanimoto_similarity_to_top",
            "central_priority",
            "n_members",
            "median_interaction_count",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    central_df["rank_central"] = range(1, len(central_df) + 1)
    return out, central_df, reference_smiles


def build_unique_df(scaf_df, args, sort_cols=None, ascending=None, na_position="last"):
    if scaf_df.empty:
        return scaf_df.copy()

    if args.unique_by_novelty:
        quality_mask = scaf_df["_quality_norm"].map(lambda q: q is not None and q >= args.min_unique_quality)
        unique_df = scaf_df.loc[quality_mask].copy()
    else:
        unique_df = scaf_df.loc[scaf_df["n_members"] <= args.max_unique_group_size].copy()

    unique_df = unique_df.loc[pd.to_numeric(unique_df["n_members"], errors="coerce") >= float(args.unique_min_members)].copy()
    novelty_floor = float(getattr(args, "unique_min_novelty", 0.0))
    if novelty_floor > 0.0 and "interaction_novelty" in unique_df.columns:
        novelty = pd.to_numeric(unique_df["interaction_novelty"], errors="coerce")
        unique_df = unique_df.loc[novelty.notna() & (novelty >= novelty_floor)].copy()

    if sort_cols is None:
        sort_cols = ["unique_priority", "interaction_novelty", "median_interaction_count", "median_score"]
    if ascending is None:
        ascending = [False, False, False, True]

    valid_cols = []
    valid_asc = []
    for col, asc in zip(sort_cols, ascending):
        if col in unique_df.columns:
            valid_cols.append(col)
            valid_asc.append(asc)

    if valid_cols:
        unique_df = unique_df.sort_values(valid_cols, ascending=valid_asc, na_position=na_position).reset_index(drop=True)
    else:
        unique_df = unique_df.reset_index(drop=True)

    unique_df["rank_unique"] = range(1, len(unique_df) + 1)
    return unique_df


def build_per_scaffold_substitution_table(task):
    scaf_id, records = task
    sdf = pd.DataFrame.from_records(records)
    if sdf.empty:
        return hash_text(scaf_id), pd.DataFrame()

    positions = sorted(
        {
            int(match.group(1))
            for sig in sdf["substitution_signature"].dropna().tolist()
            for match in re.finditer(r"R(\d+)=", sig)
        }
    )

    rows = []
    for _, row in sdf.iterrows():
        rec = {
            "mol_id": row.get("mol_id"),
            "score": row.get("score"),
            "interaction_count": row.get("interaction_count"),
            "priority_score": row.get("priority_score"),
            "cluster_id": row.get("cluster_id"),
            "substitution_signature": row.get("substitution_signature"),
            "smiles": row.get("smiles"),
        }
        pos_map = row.get("substitution_map") if isinstance(row.get("substitution_map"), dict) else {}
        for pos in positions:
            frags = pos_map.get(pos - 1, ())
            rec[f"R{pos}"] = ".".join(frags) if frags else ""
        rows.append(rec)

    tdf = pd.DataFrame(rows)
    tdf = tdf.sort_values(["priority_score", "interaction_count", "score"], ascending=[False, False, True])
    return hash_text(scaf_id), tdf


def draw_scaffold_with_positions(scaf, positions, out_png, legend=None):
    mol = depiction_mol(scaf)
    rdDepictor.Compute2DCoords(mol)
    for pos in positions:
        if pos < mol.GetNumAtoms():
            atom = mol.GetAtomWithIdx(pos)
            atom.SetProp("atomNote", f"R{pos + 1}")
    drawer = rdMolDraw2D.MolDraw2DCairo(900, 500)
    dopts = drawer.drawOptions()
    dopts.addStereoAnnotation = False
    dopts.explicitMethyl = False
    dopts.fixedBondLength = 35
    highlight_atoms = list(sorted(set(positions)))
    drawer.DrawMolecule(mol, legend=legend or "", highlightAtoms=highlight_atoms, highlightBonds=[])
    drawer.FinishDrawing()
    with open(out_png, "wb") as fh:
        fh.write(drawer.GetDrawingText())


def draw_scaffold_panel(scaf, positions, member_df, out_png, top_n=4, global_reference_smiles=None):
    core = depiction_mol(scaf)
    rdDepictor.Compute2DCoords(core)
    for pos in positions:
        if pos < core.GetNumAtoms():
            core.GetAtomWithIdx(pos).SetProp("atomNote", f"R{pos + 1}")

    var_label = ", ".join([f"R{pos + 1}" for pos in sorted(set(positions))]) if positions else "none"
    mols = [core]
    legends = [f"Central motif\nVariable R-map: {var_label}"]
    highlight_atom_lists = [list(sorted(set(positions)))]

    rep_df = representative_ranked_subset(member_df, top_n=top_n)
    for _, row in rep_df.iterrows():
        smi = row.get("smiles")
        mid = str(row.get("mol_id", ""))
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol = aligned_depiction_mol(mol, reference=(global_reference_smiles or scaf))
        mols.append(mol)
        legends.append(f"{mid}\nscore={row.get('score')} | int={row.get('interaction_count')}")
        ovr = row.get("overall_score")
        ovr_str = f"{ovr:.2f}" if ovr is not None and not (isinstance(ovr, float) and pd.isna(ovr)) else "—"
        dl = row.get("druglike_score")
        dl_str = f"{dl:.2f}" if dl is not None and not (isinstance(dl, float) and pd.isna(dl)) else "—"
        sig_short = filter_sig_to_positions(row.get("substitution_signature", ""), positions)
        if len(sig_short) > 90:
            sig_short = sig_short[:87] + "..."
        legends[-1] = (
            f"{mid}\nscr={row.get('score')} | int={row.get('interaction_count')} | dl={dl_str} | ovr={ovr_str}"
            f"\n{sig_short}"
        )
        highlight_atom_lists.append([])

    if len(mols) == 1:
        return

    n_cols = min(3, len(mols))
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=n_cols,
        subImgSize=(340, 280),
        legends=legends,
        highlightAtomLists=highlight_atom_lists,
        useSVG=False,
    )
    img.save(out_png)


def build_scaffold_summary_row(task, *, eprint=None):
    scaf_id, records, scaf_img_dir, file_prefix = task
    sdf = pd.DataFrame.from_records(records)
    if sdf.empty:
        return None

    ex = sdf["exact_scaffold_smiles"].dropna().iloc[0] if sdf["exact_scaffold_smiles"].notna().any() else None
    gen = sdf["generic_scaffold_smiles"].dropna().iloc[0] if sdf["generic_scaffold_smiles"].notna().any() else None
    n_members = len(sdf)

    med_int = float(sdf["interaction_count"].dropna().median()) if sdf["interaction_count"].notna().any() else None
    best_int = float(sdf["interaction_count"].dropna().max()) if sdf["interaction_count"].notna().any() else None
    med_score = float(sdf["score"].dropna().median()) if sdf["score"].notna().any() else None
    best_score = float(sdf["score"].dropna().min()) if sdf["score"].notna().any() else None
    med_overall = float(sdf["overall_score"].dropna().median()) if "overall_score" in sdf.columns and sdf["overall_score"].notna().any() else None

    n_unique_sig = int(sdf["substitution_signature"].nunique())
    panel_subset = representative_ranked_subset(sdf, top_n=4)
    var_positions = variable_positions_from_members(panel_subset)
    dominant_patterns = variable_substitution_summary(panel_subset, var_positions, top_n=3)
    rep_subset = representative_ranked_subset(sdf, top_n=10)
    rep_ids = rep_subset["mol_id"].tolist()
    representative_ids = ", ".join([str(x) for x in rep_ids])
    representative_smiles = rep_subset.iloc[0].get("smiles") if not rep_subset.empty else None

    core_png_b64 = None
    if ex:
        try:
            core_png_b64 = mol_png_base64(Chem.MolFromSmiles(ex), size=(220, 160), legend="", reference=ex)
        except Exception:
            core_png_b64 = None

    png_rel = None
    if ex:
        try:
            scaf = Chem.MolFromSmiles(ex)
            fname = prefixed_output_name(file_prefix, f"scaffold_{hash_text(scaf_id)}.png")
            fpath = os.path.join(scaf_img_dir, fname)
            draw_scaffold_with_positions(scaf, var_positions, fpath, legend=f"n={n_members}")
            png_rel = os.path.join("scaffold_images", fname)
        except Exception as exc:
            if eprint is not None:
                eprint(f"Warning: could not draw scaffold {scaf_id}: {exc}")

    scaffold_fsp3 = None
    scaffold_aliphatic_rings = None
    if ex:
        try:
            scaffold_mol = Chem.MolFromSmiles(ex)
            if scaffold_mol is not None:
                scaffold_fsp3 = float(rdMolDescriptors.CalcFractionCSP3(scaffold_mol))
                scaffold_aliphatic_rings = int(rdMolDescriptors.CalcNumAliphaticRings(scaffold_mol))
        except Exception:
            pass

    return {
        "scaffold_id": scaf_id,
        "scaffold_label": str(scaf_id)[:60],
        "scaffold_name": None,
        "exact_scaffold_smiles": ex,
        "generic_scaffold_smiles": gen,
        "n_members": n_members,
        "n_unique_substitution_signatures": n_unique_sig,
        "median_interaction_count": med_int,
        "best_interaction_count": best_int,
        "median_score": med_score,
        "best_score": best_score,
        "median_overall_score": med_overall,
        "dominant_substitution_patterns": dominant_patterns,
        "representative_ids": representative_ids,
        "representative_smiles": representative_smiles,
        "core_png_b64": core_png_b64,
        "scaffold_png": png_rel,
        "scaffold_panel_png": None,
        "variable_positions": var_positions,
        "interaction_novelty": None,
        "scaffold_fsp3": scaffold_fsp3,
        "scaffold_aliphatic_rings": scaffold_aliphatic_rings,
    }


def add_scaffold_stats(scaf_df):
    scaf_df["substitution_diversity"] = scaf_df["n_unique_substitution_signatures"] / scaf_df["n_members"].clip(lower=1)
    return scaf_df


def compute_interaction_novelty(scaf_df, mol_df):
    contact_cols = [col for col in mol_df.columns if col.endswith("_contact")]
    scaffold_ids = scaf_df["scaffold_id"].tolist()

    if not contact_cols:
        vals = normalize_series(scaf_df["substitution_diversity"].tolist(), higher_is_better=True)
        return {sid: (val if val is not None else 0.0) for sid, val in zip(scaffold_ids, vals)}

    profiles = {}
    for sid in scaffold_ids:
        sub = mol_df[mol_df["scaffold_id"] == sid][contact_cols].fillna(0.0)
        if sub.empty:
            profiles[sid] = set()
        else:
            mean_contacts = sub.mean(axis=0)
            profiles[sid] = set(col for col, val in mean_contacts.items() if val > 0)

    novelty = {}
    for sid in scaffold_ids:
        current = profiles[sid]
        if not current:
            novelty[sid] = 0.0
            continue
        max_sim = 0.0
        for other_sid in scaffold_ids:
            if other_sid == sid:
                continue
            other = profiles[other_sid]
            if not other:
                continue
            intersection = len(current & other)
            union = len(current | other)
            sim = intersection / union if union > 0 else 0.0
            if sim > max_sim:
                max_sim = sim
        novelty[sid] = round(1.0 - max_sim, 4)
    return novelty


def _scaffold_dataframe_from_rows(scaf_rows):
    scaf_columns = [
        "scaffold_id",
        "scaffold_label",
        "scaffold_name",
        "exact_scaffold_smiles",
        "generic_scaffold_smiles",
        "n_members",
        "n_unique_substitution_signatures",
        "median_interaction_count",
        "best_interaction_count",
        "median_score",
        "best_score",
        "median_overall_score",
        "dominant_substitution_patterns",
        "representative_ids",
        "representative_smiles",
        "core_png_b64",
        "scaffold_png",
        "scaffold_panel_png",
        "variable_positions",
        "interaction_novelty",
        "scaffold_fsp3",
        "scaffold_aliphatic_rings",
    ]
    scaf_df = pd.DataFrame(scaf_rows, columns=scaf_columns)
    if scaf_df.empty:
        for col, dtype in [
            ("substitution_diversity", float),
            ("central_priority", float),
            ("unique_priority", float),
            ("tanimoto_similarity_to_top", float),
            ("tanimoto_distance_to_top", float),
            ("central_rerank_score", float),
            ("high_distance_central", bool),
            ("_quality_norm", float),
            ("rank_central", int),
            ("rank_unique", int),
        ]:
            scaf_df[col] = pd.Series(dtype=dtype)
    else:
        scaf_df = add_scaffold_stats(scaf_df)
    return scaf_df


def initialize_scaffold_analysis(scaf_rows, report_mol_df, args, include_unique=True):
    scaf_df = _scaffold_dataframe_from_rows(scaf_rows)
    if scaf_df.empty:
        return scaf_df, scaf_df.copy(), scaf_df.copy(), None

    size_norm = normalize_series(scaf_df["n_members"].tolist(), higher_is_better=True)
    div_norm = normalize_series(scaf_df["substitution_diversity"].tolist(), higher_is_better=True)
    med_int_norm = normalize_series(scaf_df["median_interaction_count"].tolist(), higher_is_better=True)
    med_score_norm = normalize_series(scaf_df["median_score"].tolist(), higher_is_better=False)

    quality_norm = [weighted_present([(i, 0.60), (s, 0.40)]) for i, s in zip(med_int_norm, med_score_norm)]

    central_priority = []
    unique_priority = []
    for idx in range(len(scaf_df)):
        rarity = None if size_norm[idx] is None else (1.0 - size_norm[idx])
        cp = weighted_present([(size_norm[idx], 0.45), (quality_norm[idx], 0.35), (div_norm[idx], 0.20)])
        up = weighted_present([(rarity, 0.45), (quality_norm[idx], 0.35), (div_norm[idx], 0.20)])
        central_priority.append(cp if cp is not None else -1.0)
        unique_priority.append(up if up is not None else -1.0)

    scaf_df["central_priority"] = central_priority
    scaf_df["unique_priority"] = unique_priority
    scaf_df, central_df, global_reference_smiles = annotate_similarity_and_rank_central(scaf_df, args.min_group_size)
    scaf_df["_quality_norm"] = quality_norm

    novelty_map = compute_interaction_novelty(scaf_df, report_mol_df)
    scaffold_ids_list = scaf_df["scaffold_id"].tolist()
    novelty_list = [novelty_map.get(sid, 0.0) for sid in scaffold_ids_list]
    scaf_df["interaction_novelty"] = novelty_list

    struct_div_norm = normalize_series(scaf_df["scaffold_fsp3"].fillna(0.0).tolist(), higher_is_better=True)
    unique_priority_novelty = []
    for idx in range(len(scaf_df)):
        nov = novelty_list[idx]
        up = weighted_present([
            (nov, 0.40),
            (quality_norm[idx], 0.30),
            (struct_div_norm[idx], 0.20),
            (div_norm[idx], 0.10),
        ])
        unique_priority_novelty.append(up if up is not None else -1.0)
    scaf_df["unique_priority"] = unique_priority_novelty

    if include_unique:
        unique_df = build_unique_df(scaf_df, args)
    else:
        unique_df = scaf_df.head(0).copy()
    return scaf_df, central_df, unique_df, global_reference_smiles


def assign_scaffold_names_and_rerank(scaf_df, args):
    if scaf_df.empty:
        return scaf_df.copy(), scaf_df.copy(), None

    out = scaf_df.sort_values(
        ["central_rerank_score", "central_priority", "n_members"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    out["scaffold_name"] = [f"SCF-{idx + 1:03d}" for idx in range(len(out))]
    out, central_df, global_reference_smiles = annotate_similarity_and_rank_central(out, args.min_group_size)
    global_reference_smiles = select_depiction_reference_smiles(out, preferred_scaffold_name="SCF-001")
    return out, central_df, global_reference_smiles