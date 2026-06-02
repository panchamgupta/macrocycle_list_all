import base64
import html
import json
import os
import re
import shutil

import pandas as pd
from rdkit import Chem

from docking_pose_visualizer_block import build_docking_pose_visualizer_js
from scaffold_summary_helpers import filter_sig_to_positions, mol_png_base64_from_smiles, mol_png_base64_from_smiles_with_status, representative_ranked_subset
from shared_utils import hash_text, safe_float


_REPORT_HELPERS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPORT_ASSET_VENDOR_DIR = os.path.join(_REPORT_HELPERS_DIR, "report_assets", "vendor")

# Reference ligand indices are offset by this amount to never collide with real pose indices.
_REF_LIGAND_INDEX_OFFSET = 1_000_000


def parse_ref_ligand_sdf(sdf_path):
    """Parse a multi-entry SDF into a list of reference ligand dicts.

    Each dict has: title (str), sdf_block (str), img_b64 (str|None), idx (int).
    Title comes from mol block line 1. Indices start at _REF_LIGAND_INDEX_OFFSET.
    """
    if not sdf_path or not os.path.exists(sdf_path):
        return []
    with open(sdf_path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    results = []
    raw_blocks = re.split(r"\$\$\$\$", text)
    entry_idx = 0
    for block in raw_blocks:
        stripped = block.strip()
        if not stripped:
            continue
        lines = stripped.splitlines()
        title = lines[0].strip() if lines else ""
        if not title:
            title = f"Ref {entry_idx + 1}"
        sdf_block = stripped + "\n$$$$\n"
        img_b64 = None
        try:
            mol = Chem.MolFromMolBlock(stripped, removeHs=True, sanitize=True)
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                img_b64 = mol_png_base64_from_smiles(smiles, size=(280, 180), legend=title)
        except Exception:
            pass
        results.append(
            {
                "title": title,
                "sdf_block": sdf_block,
                "img_b64": img_b64,
                "idx": _REF_LIGAND_INDEX_OFFSET + entry_idx,
            }
        )
        entry_idx += 1
    return results


def ref_ligand_tile_html(entry):
    """Render a single reference ligand tile with overlay checkbox and View Pose button."""
    idx = entry["idx"]
    title = entry["title"]
    title_esc = html.escape(title)
    title_attr = html.escape(title, quote=True)
    title_js = json.dumps(title)  # safe JS string literal
    parts = [
        "<div class='moltile ref-ligand-tile' "
        "style='border:2px solid #c84c09;background:#fff9f6;'>"
    ]
    # Overlay checkbox
    parts.append(
        "<label class='smalltxt' style='display:flex;align-items:center;gap:6px;margin-bottom:4px;cursor:pointer;'>"
        f"<input type='checkbox' class='pose-overlay-toggle' "
        f"data-mol-index='{idx}' data-mol-label='{title_attr}'/>"
        "Overlay in viewer"
        "</label>"
    )
    # 2D image
    img_b64 = entry.get("img_b64")
    if img_b64:
        parts.append(f"<img src='data:image/png;base64,{img_b64}' decoding='async'/>")
    # Title + badge
    parts.append(
        f"<div><b>{title_esc}</b> "
        "<span style='display:inline-block;background:#c84c09;color:#fff;"
        "border-radius:999px;padding:1px 7px;font-size:10px;font-weight:700;margin-left:4px;'>"
        "Reference</span></div>"
    )
    # View Pose button
    parts.append(
        "<div style='margin-top:6px;'>"
        "<button type='button' "
        "style='background:#c84c09;border:none;color:#fff;border-radius:6px;"
        "padding:5px 12px;cursor:pointer;font-size:12px;font-weight:600;' "
        f"onclick=\"if(typeof _openPoseVisualizerWindow==='function'){{_openPoseVisualizerWindow({idx},{title_js});}}else if(typeof _showPoseFromIndex==='function'){{_showPoseFromIndex({idx},{title_js});}}else{{_renderPoseFromIndex({idx},{title_js});}}\">&#9654; View Pose</button>"
        "</div>"
    )
    parts.append("</div>")
    return "".join(parts)


def parse_mol2_serial_map(mol2_text):
    """Parse a Tripos MOL2 string and return {str(serial): 'RESN RESI'} for each atom.

    MOL2 @<TRIPOS>ATOM line fields (1-indexed, space-separated):
      1=serial  2=atom_name  3=x  4=y  5=z  6=atom_type  7=subst_id  8=subst_name  9=charge
    subst_name is like 'ALA195' (3-letter AA code + PDB residue number).
    """
    result = {}
    in_atom = False
    for line in mol2_text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith('@<TRIPOS>ATOM'):
            in_atom = True
            continue
        if stripped.startswith('@<TRIPOS>') and in_atom:
            break
        if in_atom and stripped and not stripped.startswith('#'):
            parts = stripped.split()
            if len(parts) >= 8:
                try:
                    serial = int(parts[0])
                    subst_name = parts[7]  # e.g. 'ALA195', 'HOH401'
                    m = re.match(r'^([A-Za-z]+)(\d+)$', subst_name)
                    if m:
                        resn = m.group(1).upper()
                        resi = int(m.group(2))
                        result[str(serial)] = resn + ' ' + str(resi)
                except (ValueError, IndexError):
                    pass
    return result


def parse_pdb_serial_map(pdb_text):
    """Parse a PDB string and return {str(serial): 'RESN RESI'} for ATOM/HETATM records."""
    result = {}
    for line in pdb_text.splitlines():
        rec = line[:6].strip().upper()
        if rec not in ('ATOM', 'HETATM'):
            continue
        try:
            serial = int(line[6:11].strip())
            resn = line[17:20].strip().upper()
            resi = int(line[22:26].strip())
            if resn and resi is not None:
                result[str(serial)] = resn + ' ' + str(resi)
        except (ValueError, IndexError):
            pass
    return result
_RDKIT_VENDOR_JS = os.path.join(_REPORT_ASSET_VENDOR_DIR, "rdkit", "RDKit_minimal.js")
_RDKIT_VENDOR_WASM = os.path.join(_REPORT_ASSET_VENDOR_DIR, "rdkit", "RDKit_minimal.wasm")
_RDKIT_LOCAL_JS = os.path.join(_REPORT_HELPERS_DIR, "report_assets", "rdkit", "RDKit_minimal.js")
_RDKIT_LOCAL_WASM = os.path.join(_REPORT_HELPERS_DIR, "report_assets", "rdkit", "RDKit_minimal.wasm")


def _resolve_rdkit_asset_paths():
    js_candidates = [
        _RDKIT_VENDOR_JS,
        _RDKIT_LOCAL_JS,
    ]
    wasm_candidates = [
        _RDKIT_VENDOR_WASM,
        _RDKIT_LOCAL_WASM,
    ]
    js_path = next((path for path in js_candidates if os.path.exists(path)), None)
    wasm_path = next((path for path in wasm_candidates if os.path.exists(path)), None)
    missing = []
    if js_path is None:
        missing.append(_RDKIT_VENDOR_JS)
    if wasm_path is None:
        missing.append(_RDKIT_VENDOR_WASM)
    if missing:
        raise FileNotFoundError(f"Missing structure-search asset(s): {missing}")
    return js_path, wasm_path


def canonicalize_smiles(smiles):
    if smiles is None or pd.isna(smiles):
        return ""
    text = str(smiles).strip()
    if not text:
        return ""
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def canonicalize_sdf_block(sdf_block):
    if not sdf_block:
        return ""
    try:
        mol = Chem.MolFromMolBlock(str(sdf_block), removeHs=True, sanitize=True)
    except Exception:
        mol = None
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return ""


def canonicalize_row_structure(row, pose_sdf_by_index=None):
    pose_sdf_by_index = pose_sdf_by_index or {}
    mol_index = row.get("mol_index")
    if mol_index is not None and not pd.isna(mol_index):
        sdf_block = pose_sdf_by_index.get(mol_index)
        if sdf_block is None:
            sdf_block = pose_sdf_by_index.get(str(mol_index))
        canonical_from_sdf = canonicalize_sdf_block(sdf_block)
        if canonical_from_sdf:
            return canonical_from_sdf
    return canonicalize_smiles(row.get("smiles"))


def build_structure_search_entries(mol_df, allowed_scaffold_names=None, pose_sdf_by_index=None):
    allowed = None if allowed_scaffold_names is None else {str(name) for name in allowed_scaffold_names if str(name).strip()}
    entries = []
    seen = set()
    for _, row in mol_df.iterrows():
        scaffold_name = str(row.get("scaffold_name", "") or "").strip()
        if allowed is not None and scaffold_name not in allowed:
            continue
        canonical_smiles = canonicalize_row_structure(row, pose_sdf_by_index=pose_sdf_by_index)
        if not canonical_smiles:
            continue
        mol_id = str(row.get("mol_id", "") or "").strip()
        if not mol_id:
            mol_index = row.get("mol_index")
            mol_id = f"mol-{mol_index}" if mol_index is not None and not pd.isna(mol_index) else f"mol-{len(entries) + 1}"
        key = (scaffold_name, mol_id, canonical_smiles)
        if key in seen:
            continue
        seen.add(key)
        exact_no_stereo = canonical_smiles
        try:
            exact_mol = Chem.MolFromSmiles(canonical_smiles)
            if exact_mol is not None:
                exact_no_stereo = Chem.MolToSmiles(exact_mol, isomericSmiles=False)
        except Exception:
            pass
        entries.append(
            {
                "scaffold": scaffold_name,
                "mol_id": mol_id,
                "smiles": canonical_smiles,
                "exact_canonical": canonical_smiles,
                "exact_canonical_nostereo": exact_no_stereo,
            }
        )
    return entries


def ensure_structure_search_assets(outdir):
    rdkit_js_src, rdkit_wasm_src = _resolve_rdkit_asset_paths()

    asset_root = os.path.join(outdir, "report_assets")
    rdkit_root = os.path.join(asset_root, "rdkit")
    os.makedirs(rdkit_root, exist_ok=True)

    js_dst = os.path.join(rdkit_root, "RDKit_minimal.js")
    wasm_dst = os.path.join(rdkit_root, "RDKit_minimal.wasm")

    if os.path.abspath(rdkit_js_src) != os.path.abspath(js_dst):
        shutil.copyfile(rdkit_js_src, js_dst)
    if os.path.abspath(rdkit_wasm_src) != os.path.abspath(wasm_dst):
        shutil.copyfile(rdkit_wasm_src, wasm_dst)

    return {
        "rdkit_js_src": "report_assets/rdkit/RDKit_minimal.js",
        "rdkit_wasm_src": "report_assets/rdkit/RDKit_minimal.wasm",
    }


def get_embedded_rdkit_wasm_b64():
    _, rdkit_wasm_src = _resolve_rdkit_asset_paths()
    with open(rdkit_wasm_src, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _image_data_url(abs_path):
    if not abs_path or not os.path.exists(abs_path):
        return ""
    ext = os.path.splitext(abs_path)[1].lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
    }.get(ext, "application/octet-stream")
    with open(abs_path, "rb") as fh:
        payload = base64.b64encode(fh.read()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def molecule_tile_html(
    row,
    show_scaffold=False,
    global_reference_smiles=None,
    enable_pose_hover=False,
    enable_overlay_checkbox=False,
    enable_member_checkbox=False,
    member_scaffold_name=None,
    member_hbond_residues=None,
    mol_props_data=None,
    prop_names_list=None,
):
    reference_smiles = global_reference_smiles
    tile_png = mol_png_base64_from_smiles(
        row.get("smiles"),
        size=(280, 180),
        legend=str(row.get("mol_id", "")),
        reference_smiles=reference_smiles,
    )
    mol_idx = row.get("mol_index")
    tile_classes = ["moltile"]
    tile_attrs = []
    if enable_pose_hover and mol_idx is not None and not pd.isna(mol_idx):
        tile_classes.append("pose-hover")
        tile_attrs.append(f"data-mol-index='{int(mol_idx)}'")
    mol_id = str(row.get("mol_id", "") or "").strip()
    if mol_id:
        tile_attrs.append(f"data-mol-id='{html.escape(mol_id, quote=True)}'")
    scaffold_name = str(member_scaffold_name or row.get("scaffold_name", "") or "").strip()
    if scaffold_name:
        tile_attrs.append(f"data-scaffold='{html.escape(scaffold_name, quote=True)}'")
    smiles = str(row.get("smiles", "") or "").strip()
    if smiles:
        tile_attrs.append(f"data-search-smiles='{html.escape(smiles, quote=True)}'")
    exact_canonical = str(row.get("search_exact_canonical", "") or row.get("exact_canonical_smiles", "") or "").strip()
    if exact_canonical:
        tile_attrs.append(f"data-search-canonical='{html.escape(exact_canonical, quote=True)}'")
    attrs_str = " ".join(tile_attrs)
    attrs_str = f" {attrs_str}" if attrs_str else ""
    parts = [f"<div class='{' '.join(tile_classes)}'{attrs_str}>"]
    if enable_overlay_checkbox and mol_idx is not None and not pd.isna(mol_idx):
        safe_mid = html.escape(str(row.get("mol_id", "")))
        parts.append(
            "<label class='smalltxt' style='display:flex;align-items:center;gap:6px;margin-bottom:4px;cursor:pointer;'>"
            f"<input type='checkbox' class='pose-overlay-toggle' data-mol-index='{int(mol_idx)}' data-mol-label='{safe_mid}'/>"
            "Overlay in viewer"
            "</label>"
        )
    if enable_member_checkbox and mol_idx is not None and not pd.isna(mol_idx):
        safe_mid = html.escape(str(row.get("mol_id", "")))
        safe_scaf = html.escape(str(member_scaffold_name or row.get("scaffold_name", "")))
        parts.append(
            "<label class='smalltxt' style='display:flex;align-items:center;gap:6px;margin-bottom:4px;cursor:pointer;'>"
            f"<input type='checkbox' class='member-export-toggle' data-mol-index='{int(mol_idx)}' data-mol-label='{safe_mid}' data-scaffold='{safe_scaf}'/>"
            "Select member for SDF"
            "</label>"
        )
    if tile_png:
        parts.append(f"<img src='data:image/png;base64,{tile_png}' decoding='async'/>")
    parts.append(f"<div><b>{html.escape(str(row.get('mol_id', '')))}</b></div>")
    if show_scaffold and row.get("scaffold_name"):
        parts.append(f"<div class='smalltxt'><b>Scaffold:</b> {html.escape(str(row.get('scaffold_name', '')))}</div>")
    dl = row.get("druglike_score")
    dl_str = f"{dl:.2f}" if dl is not None and not (isinstance(dl, float) and pd.isna(dl)) else "—"
    ovr = row.get("overall_score")
    ovr_str = f"{ovr:.3f}" if ovr is not None and not (isinstance(ovr, float) and pd.isna(ovr)) else "—"
    parts.append(
        f"<div class='smalltxt'>"
        f"score: {row.get('score')} | "
        f"interactions: {row.get('interaction_count')} | "
        f"druglike: {dl_str} | "
        f"overall: {ovr_str} | "
        f"rotB: {row.get('rot_bonds')} | TPSA: {row.get('tpsa')}"
        f"</div>"
    )
    torsion_angle = safe_float(row.get("torsion_angle"))
    if torsion_angle is not None:
        parts.append(f"<div class='smalltxt'>Torsion: {torsion_angle:.2f} deg</div>")
    # Add the 5 special properties if available.
    if mol_props_data is not None and prop_names_list is not None:
        mol_id = str(row.get("mol_id", ""))
        if mol_id in mol_props_data:
            props_arr = mol_props_data[mol_id]  # list indexed by prop position
            target_props = ["GS_LogD", "GS_Sol_74_linear", "MW", "RingCount", "FractionCSP3"]
            target_labels = ["LogD", "Sol(7.4)", "MW", "Rings", "Fsp3"]
            prop_vals = []
            for p, lbl in zip(target_props, target_labels):
                if p in prop_names_list:
                    idx = prop_names_list.index(p)
                    if idx < len(props_arr) and props_arr[idx] is not None:
                        val = props_arr[idx]
                        val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                        prop_vals.append(f"{lbl}: {val_str}")
            if prop_vals:
                parts.append(
                    f"<div class='smalltxt' style='color:#1f4f7a;margin-top:2px;'>"
                    f"{' | '.join(prop_vals)}"
                    f"</div>"
                )
    if member_hbond_residues:
        residue_badges = " ".join(
            [
                f"<span style='display:inline-block;background:#eef8ff;color:#1f4f7a;border:1px solid #bfdcf5;border-radius:999px;padding:2px 8px;font-size:11px;font-weight:600;'>H-bond {html.escape(str(residue))}</span>"
                for residue in member_hbond_residues
            ]
        )
        parts.append(
            "<div class='smalltxt' style='margin-top:6px;display:flex;gap:6px;flex-wrap:wrap;align-items:center;'><b>H-bond residues:</b> "
            f"{residue_badges}</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def html_img_from_b64(b64, alt, width=180):
    if not b64:
        return ""
    return f"<img src='data:image/png;base64,{b64}' alt='{html.escape(str(alt), quote=True)}' decoding='async' style='max-width:{width}px;width:100%;height:auto;' />"


def build_idea_cards(
    df,
    card_type,
    with_checkboxes=False,
    highlighted_scaffold_names=None,
    include_links=False,
    scaffold_hbond_map=None,
    top_per_scaffold=None,
    scaffold_prop_ranges=None,
    scaffold_unique_member_counts=None,
    global_reference_smiles=None,
    global_reference_core_smarts=None,
):
    if df.empty:
        return "<p><i>None</i></p>"
    highlight_set = set(highlighted_scaffold_names or [])
    blocks = []
    for _, row in df.iterrows():
        sname = str(row.get("scaffold_name", ""))
        sname_js = sname.replace("\\", "\\\\").replace("'", "\\'")
        mol_index = row.get("mol_index")
        mol_id = str(row.get("mol_id") or row.get("scaffold_name") or "Molecule")
        card_click_attr = ""
        if mol_index is not None and not pd.isna(mol_index):
            midx = int(mol_index)
            mol_label_js = mol_id.replace("\\", "\\\\").replace("'", "\\'")
            card_click_attr = (
                " data-mol-index='" + str(midx) + "'"
                + " data-mol-label='" + html.escape(mol_id, quote=True) + "'"
                + " style='cursor:pointer;'"
                + " onclick=\"if(event&&event.target&&event.target.closest('button,input,select,textarea,label,a')){return;}"
                + "if(typeof _openPoseVisualizerWindow==='function'){_openPoseVisualizerWindow(" + str(midx) + ",'" + mol_label_js + "');}"
                + "else if(typeof _showPoseFromIndex==='function'){_showPoseFromIndex(" + str(midx) + ",'" + mol_label_js + "');}"
                + "else if(typeof _renderPoseFromIndex==='function'){_renderPoseFromIndex(" + str(midx) + ",'" + mol_label_js + "');}\""
            )
        card_id = f"ci-{hash_text(sname)}"
        card_classes = ["idea-card"]
        cb_html = (
            f"<label style='display:flex;align-items:center;gap:5px;cursor:pointer;'>"
            f"<input type='checkbox' class='scaf-checkbox' data-scaffold='{sname}' "
            f"onchange=\"syncCheckboxes('{sname_js}', this.checked)\" />"
            f"<span style='font-size:11px;color:#5f6d7a;'>Select</span></label>"
        ) if with_checkboxes else ""
        n_members = int(row.get("n_members", 0) or 0)
        unique_members = int((scaffold_unique_member_counts or {}).get(sname, n_members) or 0)
        all_members_cb_html = ""
        if with_checkboxes and top_per_scaffold is not None and n_members > top_per_scaffold:
            all_members_cb_html = (
                f"<label style='display:flex;align-items:center;gap:5px;cursor:pointer;font-size:11px;color:#5f6d7a;'>"
                f"<input type='checkbox' class='all-members-checkbox' data-scaffold='{sname}' />"
                f"<span class='all-members-label' data-raw-members='{n_members}' data-unique-members='{unique_members}'>"
                f"Include all {n_members} members in SDF"
                f"</span></label>"
            )
        star_html = ""
        if card_type == "Central":
            star_html = (
                f"<div class='star-wrap' data-scaffold='{sname}'>"
                f"<span class='smalltxt'>Star:</span>"
                f"<button type='button' class='star-btn' data-level='1' onclick=\"setStar('{sname_js}', 1)\">1</button>"
                f"<button type='button' class='star-btn' data-level='2' onclick=\"setStar('{sname_js}', 2)\">2</button>"
                f"<button type='button' class='star-btn' data-level='3' onclick=\"setStar('{sname_js}', 3)\">3</button>"
                f"<button type='button' class='star-btn star-clear' onclick=\"setStar('{sname_js}', 0)\">x</button>"
                f"</div>"
            )
        deactivate_html = ""
        if card_type == "Central":
            deactivate_html = (
                f"<label class='deactivate-control smalltxt' "
                f"style='display:inline-flex;align-items:center;gap:5px;margin-left:8px;cursor:pointer;'>"
                f"<input type='checkbox' class='deactivate-toggle' data-scaffold='{sname}' "
                f"onchange=\"setScaffoldDeactivated('{sname_js}', this.checked)\" />"
                "Deactivate</label>"
            )
        range_html = ""
        if card_type == "Central":
            display_props = [
                ("GS_LogD", "LogD"),
                ("GS_Sol_74_linear", "Sol(7.4)"),
                ("MW", "MW"),
                ("RingCount", "Rings"),
                ("FractionCSP3", "Fsp3"),
            ]
            range_parts = []
            scaffold_ranges = (scaffold_prop_ranges or {}).get(sname, {})
            for prop_name, label in display_props:
                prop_range = scaffold_ranges.get(prop_name)
                if not prop_range:
                    continue
                mn = prop_range.get("min")
                mx = prop_range.get("max")
                if mn is None or mx is None:
                    continue
                range_parts.append(f"{label}: {mn:.2f}\u2013{mx:.2f}")
            if range_parts:
                range_html = (
                    "<span class='central-range-inline smalltxt' style='color:#1f4f7a;'>"
                    + " | ".join(range_parts)
                    + "</span>"
                )
        drop_badge_html = ""
        if card_type == "Central":
            drop_badge_html = (
                f"<span class='central-drop-badge smalltxt' data-scaffold='{sname}' "
                f"style='display:none;color:#a94442;font-weight:700;'></span>"
            )
        blocks.append(
            f"<div class='{' '.join(card_classes)}' id='{card_id}' "
            f"data-scaffold='{sname}' data-n-members='{n_members}'{card_click_attr}>"
        )
        blocks.append(
            f"<div class='idea-head'>"
            f"<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>"
            f"{cb_html}<b>{row.get('scaffold_name')}</b>{drop_badge_html}{range_html}{star_html}{deactivate_html}</div>"
            f"<div class='smalltxt'>{card_type}</div></div>"
        )
        if all_members_cb_html:
            blocks.append(f"<div style='margin-top:4px;'>{all_members_cb_html}</div>")
        if include_links and card_type == "Central":
            blocks.append("")
        card_img_b64, aligned_to_template = mol_png_base64_from_smiles_with_status(
            row.get("smiles"),
            size=(420, 270),
            legend=mol_id,
            reference_smiles=global_reference_smiles,
            reference_core_smarts=global_reference_core_smarts,
        )
        if card_img_b64:
            blocks.append(f"<img src='data:image/png;base64,{card_img_b64}' alt='{html.escape(mol_id, quote=True)}' decoding='async' style='max-width:330px;width:100%;height:auto;' />")
        med_ovr = row.get("median_overall_score")
        med_ovr_str = f"{med_ovr:.2f}" if med_ovr is not None and not (isinstance(med_ovr, float) and pd.isna(med_ovr)) else "—"
        nov = row.get("interaction_novelty")
        nov_str = f"{nov:.2f}" if nov is not None and not (isinstance(nov, float) and pd.isna(nov)) else "—"
        fsp3 = row.get("scaffold_fsp3")
        fsp3_str = f"{fsp3:.2f}" if fsp3 is not None and not (isinstance(fsp3, float) and pd.isna(fsp3)) else "—"
        alip = row.get("scaffold_aliphatic_rings")
        alip_str = str(int(alip)) if alip is not None and not (isinstance(alip, float) and pd.isna(alip)) else "—"
        torsion_angle = safe_float(row.get("torsion_angle"))
        torsion_str = f"{torsion_angle:.2f} deg" if torsion_angle is not None else "—"
        struct_badge = ""
        if alip is not None and not (isinstance(alip, float) and pd.isna(alip)) and int(alip) > 0:
            struct_badge = " <span style='background:#d4edda;color:#155724;border-radius:4px;padding:1px 5px;font-size:10px;'>3D scaffold</span>"
        blocks.append(
            f"<div class='smalltxt'>"
            f"<b>Members:</b> <span class='central-members-count' data-raw-members='{n_members}' data-unique-members='{unique_members}'>{n_members}</span><span class='scaffold-impact-summary smalltxt' data-scaffold='{html.escape(sname, quote=True)}' style='margin-left:6px;color:#1f4f7a;'></span> | "
            f"<b>Median score:</b> {row.get('median_score')} | "
            f"<b>Median interactions:</b> {row.get('median_interaction_count')} | "
            f"<b>Median overall:</b> {med_ovr_str} | "
            f"<b>Novelty:</b> {nov_str} | "
            f"<b>Fsp3:</b> {fsp3_str} | "
            f"<b>Aliph. rings:</b> {alip_str} | "
            f"<b>Torsion:</b> {torsion_str}"
            f"{struct_badge}"
            f"</div>"
        )
        if card_type == "Central":
            residues = list((scaffold_hbond_map or {}).get(sname, []))
            if residues:
                residue_badges = " ".join(
                    [
                        f"<span style='display:inline-block;background:#eef8ff;color:#1f4f7a;border:1px solid #bfdcf5;border-radius:999px;padding:2px 8px;font-size:11px;font-weight:600;'>H-bond {html.escape(str(residue))}</span>"
                        for residue in residues
                    ]
                )
                blocks.append(f"<div class='smalltxt' style='margin-top:6px;display:flex;gap:6px;flex-wrap:wrap;align-items:center;'><b>H-bond residues:</b> {residue_badges}</div>")
        blocks.append("</div>")
    return "".join(blocks)


def to_html_table(df, cols):
    if df.empty:
        return "<p><i>None</i></p>"
    valid_cols = [col for col in cols if col in df.columns]
    return df[valid_cols].to_html(index=False, escape=False, classes="tbl")


def build_resi_name_map(protein_pdb_text, protein_structure_format):
    """Return {resi_str: 'Ser'} mapping residue number string to title-cased 3-letter AA name.

    Parses mol2 or pdb text (whichever is provided). Only unique residue names per
    residue number are stored; if two chains share the same number with different
    residues the first encountered wins (rare in single-chain structures).
    """
    result = {}
    if not protein_pdb_text:
        return result
    fmt = (protein_structure_format or "pdb").strip().lower()
    if fmt == "mol2":
        in_atom = False
        for line in protein_pdb_text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("@<TRIPOS>ATOM"):
                in_atom = True
                continue
            if stripped.startswith("@<TRIPOS>") and in_atom:
                break
            if in_atom and stripped and not stripped.startswith("#"):
                parts = stripped.split()
                if len(parts) >= 8:
                    try:
                        subst_name = parts[7]  # e.g. 'ALA195'
                        m = re.match(r"^([A-Za-z]+)(\d+)$", subst_name)
                        if m:
                            resn = m.group(1).upper()
                            resi_str = m.group(2)
                            if resi_str not in result:
                                result[resi_str] = resn.capitalize() if len(resn) == 1 else resn[0].upper() + resn[1:].lower()
                    except (ValueError, IndexError):
                        pass
    else:
        for line in protein_pdb_text.splitlines():
            rec = line[:6].strip().upper()
            if rec not in ("ATOM", "HETATM"):
                continue
            try:
                resn = line[17:20].strip().upper()
                resi_str = str(int(line[22:26].strip()))
                if resn and resi_str not in result:
                    result[resi_str] = resn[0].upper() + resn[1:].lower() if resn else resn
            except (ValueError, IndexError):
                pass
    return result


def build_hbond_residue_filter_data(report_mol_df, scaf_df):
    residue_cols = {}
    for col in report_mol_df.columns:
        match = re.match(r"^[A-Za-z]?(\d+)_(donor|acceptor)$", str(col))
        if not match:
            continue
        residue = match.group(1)
        residue_cols.setdefault(residue, []).append(col)

    if not residue_cols:
        return [], {}

    scaffold_name_map = dict(zip(scaf_df["scaffold_id"], scaf_df["scaffold_name"]))
    scaffold_residues = {}
    for scaffold_id, sdf in report_mol_df.groupby("scaffold_id", dropna=False):
        scaffold_name = scaffold_name_map.get(scaffold_id)
        if not scaffold_name:
            continue
        hits = []
        for residue, columns in residue_cols.items():
            present = False
            for col in columns:
                values = sdf[col] if col in sdf.columns else pd.Series(dtype=object)
                if values.empty:
                    continue
                normalized = values.fillna("").astype(str).str.strip()
                if normalized.ne("").any():
                    present = True
                    break
            if present:
                hits.append(residue)
        if hits:
            scaffold_residues[scaffold_name] = sorted(hits, key=lambda text: int(text))

    residue_options = sorted(residue_cols.keys(), key=lambda text: int(text))
    return residue_options, scaffold_residues


def extract_hbond_residues_from_row(row):
    residues = []
    for col, value in row.items():
        if not col.startswith("hbond_to_residue_"):
            continue
        try:
            residue = int(col.rsplit("_", 1)[-1])
        except Exception:
            continue
        try:
            present = float(value)
        except Exception:
            present = 0.0
        if present > 0:
            residues.append(str(residue))
    return sorted(set(residues), key=lambda text: int(text))

def build_central_card_payload(
    central_df,
    highlighted_scaffold_names=None,
    scaffold_export_data=None,
    scaffold_hbond_map=None,
    top_per_scaffold=None,
    mol_props_data=None,
    scaffold_mol_map=None,
    prop_names_ordered=None,
    scaffold_unique_member_counts=None,
    global_reference_smiles=None,
    global_reference_core_smarts=None,
):
    payload = []
    highlight_set = set(highlighted_scaffold_names or [])
    scaffold_prop_ranges = _compute_scaffold_prop_ranges(
        mol_props_data,
        scaffold_mol_map,
        prop_names_ordered or [],
    )
    for order, _ in enumerate(central_df.index):
        row_df = central_df.iloc[[order]]
        row = row_df.iloc[0]
        scaffold_name = str(row.get("scaffold_name", "") or "")
        payload.append(
            {
                "scaffold": scaffold_name,
                "order": order,
                "nMembers": int(row.get("n_members", 0) or 0),
                "isTop15": scaffold_name in highlight_set,
                "isHighDistance": bool(row.get("high_distance_central", False)),
                "propRanges": scaffold_prop_ranges.get(scaffold_name, {}),
                "html": build_idea_cards(
                    row_df,
                    "Central",
                    with_checkboxes=bool(scaffold_export_data),
                    highlighted_scaffold_names=highlighted_scaffold_names,
                    include_links=False,
                    scaffold_hbond_map=scaffold_hbond_map,
                    top_per_scaffold=top_per_scaffold,
                    scaffold_prop_ranges=scaffold_prop_ranges,
                    scaffold_unique_member_counts=scaffold_unique_member_counts,
                    global_reference_smiles=global_reference_smiles,
                    global_reference_core_smarts=global_reference_core_smarts,
                ),
            }
        )
    return payload


def build_scaffold_unique_member_counts(mol_df, pose_sdf_by_index=None):
    """Return {scaffold_name: unique_canonical_member_count} using full-molecule canonical smiles."""
    pose_sdf_by_index = pose_sdf_by_index or {}
    by_scaffold = {}
    for _, row in mol_df.iterrows():
        scaffold_name = str(row.get("scaffold_name", "") or "").strip()
        if not scaffold_name:
            continue
        by_scaffold.setdefault(scaffold_name, set())
        canonical = canonicalize_row_structure(row, pose_sdf_by_index=pose_sdf_by_index)
        if not canonical:
            mol_id = str(row.get("mol_id", "") or "").strip()
            mol_index = row.get("mol_index")
            canonical = mol_id or (f"mol-{int(mol_index)}" if mol_index is not None and not pd.isna(mol_index) else "")
        if canonical:
            by_scaffold[scaffold_name].add(canonical)
    return {name: len(items) for name, items in by_scaffold.items()}


# ── Properties Panel ────────────────────────────────────────────────────────

_PROP_DISPLAY_LABELS = {
    "GS_LogD": "LogD",
    "GS_Sol_74_linear": "Solubility (pH 7.4)",
    "GS_CACO2_A2B_10_linear": "Caco-2 A\u2192B (10\u03bcM)",
    "GS_CACO2_B2A_10_linear": "Caco-2 B\u2192A (10\u03bcM)",
    "GS_HP_Free_LT_linear": "Human Plasma Free (LT)",
    "GS_CACO2_A2B_1_linear": "Caco-2 A\u2192B (1\u03bcM)",
    "GS_CACO2_B2A_1_linear": "Caco-2 B\u2192A (1\u03bcM)",
    "GS_HP_Free_linear": "Human Plasma Free",
    "GS_Pred_Cl_HLM_linear": "Predicted Cl HLM",
    "GS_MDCK_linear": "MDCK Permeability",
    "GS_RED_HP_linear": "RED Human Plasma",
    "interaction_count": "Interaction Count",
    "MW": "Mol Weight",
    "cLogP": "cLogP",
    "TPSA": "TPSA",
    "HBD": "HB Donors",
    "HBA": "HB Acceptors",
    "RotBonds": "Rotatable Bonds",
    "HeavyAtoms": "Heavy Atoms",
    "FormalCharge": "Formal Charge",
    "RingCount": "Ring Count",
    "FractionCSP3": "Fsp3",
    "torsion_angle": "Torsion (deg)",
}


def _compute_prop_stats(mol_props_data, prop_names):
    """Return {prop_name: {min, max, mean, stdev, n}} from mol_props_data arrays."""
    stats = {}
    for i, p in enumerate(prop_names):
        vals = [arr[i] for arr in mol_props_data.values() if arr and arr[i] is not None]
        if vals:
            n = len(vals)
            mn = min(vals)
            mx = max(vals)
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / max(1, n - 1) if n > 1 else 0.0
            stats[p] = {
                "min": round(mn, 4),
                "max": round(mx, 4),
                "mean": round(mean, 4),
                "stdev": round(var ** 0.5, 4),
                "n": n,
            }
        else:
            stats[p] = {"min": None, "max": None, "mean": None, "stdev": None, "n": 0}
    return stats


def _build_prop_panel_html(prop_names, stats, prop_labels=None):
    """Return the full HTML for the Molecule Properties panel section."""
    label = dict(_PROP_DISPLAY_LABELS)
    label.update(prop_labels or {})

    # Stats table rows.
    rows_html = ""
    for p in prop_names:
        s = stats.get(p, {})
        lbl = html.escape(label.get(p, p))
        mn = f"{s['min']:.4g}" if s["min"] is not None else "—"
        mean = f"{s['mean']:.4g}" if s["mean"] is not None else "—"
        mx = f"{s['max']:.4g}" if s["max"] is not None else "—"
        sd = f"{s['stdev']:.4g}" if s["stdev"] is not None else "—"
        n = s.get("n", 0)
        rows_html += (
            f"<tr><td><b>{lbl}</b></td>"
            f"<td style='text-align:right;color:#556'>{n:,}</td>"
            f"<td style='text-align:right;'>{mn}</td>"
            f"<td style='text-align:right;'>{mean}</td>"
            f"<td style='text-align:right;'>{mx}</td>"
            f"<td style='text-align:right;'>{sd}</td></tr>"
        )

    # Histogram select options.
    opts_html = "".join(
        f"<option value='{p}'>{html.escape(label.get(p, p))}</option>"
        for p in prop_names
    )
    # Correlation selects — default Y to second property.
    corr_opts = opts_html
    second_prop = prop_names[1] if len(prop_names) > 1 else (prop_names[0] if prop_names else "")
    corr_y_opts = "".join(
        f"<option value='{p}' {'selected' if p == second_prop else ''}>{html.escape(label.get(p, p))}</option>"
        for p in prop_names
    )

    panel = (
        "<section class='panel collapsible' id='props-panel'>"
        "<h2 ondblclick=\"this.closest('.panel').classList.toggle('collapsed')\">Molecule Properties</h2>"
        "<p class='smalltxt'>ADME and physico-chemical properties for all molecules. "
        "Use <b>Histogram &amp; Filter</b> to apply range filters — molecule cards in Molecule List "
        "are hidden when they do not pass all active filters (AND logic). "
        "Deep-dive molecule tiles are also hidden individually. "
        "<b>Tip:</b> double-click the section title to collapse or expand this panel.</p>"
        # Active filter banner (hidden until a filter is applied).
        "<div id='prop-active-filters' style='display:none;'>"
        "<span><strong>Active Filters:</strong></span>"
        "<span id='prop-filter-chips'></span>"
        "<button type='button' onclick='_clearAllPropFilters()' "
        "style='margin-left:auto;background:#fff;border:1px solid #cca;border-radius:6px;"
        "padding:3px 10px;cursor:pointer;font-size:12px;color:#665;'>"
        "Clear All Filters</button>"
        "</div>"
        # Tab bar.
        "<div class='prop-tab-bar'>"
        "<button type='button' class='prop-tab-btn active' data-tab='stats' "
        "onclick='_showPropTab(\"stats\")'>Summary Stats</button>"
        "<button type='button' class='prop-tab-btn' data-tab='hist' "
        "onclick='_showPropTab(\"hist\")'>Histogram &amp; Filter</button>"
        "<button type='button' class='prop-tab-btn' data-tab='box' "
        "onclick='_showPropTab(\"box\")'>Box Plot</button>"
        "<button type='button' class='prop-tab-btn' data-tab='corr' "
        "onclick='_showPropTab(\"corr\")'>Correlation Plot</button>"
        "</div>"
        # Stats tab.
        "<div id='prop-tab-stats' class='prop-tab-content'>"
        "<table class='prop-stats-table'>"
        "<thead><tr><th>Property</th><th>N</th><th>Min</th><th>Mean</th><th>Max</th><th>Std Dev</th></tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
        "</div>"
        # Histogram tab.
        "<div id='prop-tab-hist' class='prop-tab-content' style='display:none;'>"
        "<div style='display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:8px;'>"
        f"<label style='font-size:13px;'>Property: <select id='prop-hist-select' onchange='_updatePropHistogram()'>{opts_html}</select></label>"
        "</div>"
        "<div id='prop-histogram-chart' style='height:240px;width:100%;min-width:0;'></div>"
        "<div class='prop-hist-controls'>"
        "<div class='prop-range-field'>Min: <input type='number' id='prop-range-min' step='any' /></div>"
        "<div class='prop-range-field'>Max: <input type='number' id='prop-range-max' step='any' /></div>"
        "<button type='button' onclick='_applyPropRangeFilter()' "
        "style='background:#0b6e4f;color:#fff;border:none;border-radius:6px;padding:5px 14px;cursor:pointer;font-size:13px;'>"
        "Apply Filter</button>"
        "<button type='button' id='prop-hist-reset-btn' onclick='_resetCurrentPropFilter()' "
        "style='background:#fff;border:1px solid #cdd;border-radius:6px;padding:5px 12px;cursor:pointer;font-size:13px;'>"
        "Reset This Filter</button>"
        "</div>"
        "<div id='prop-plotly-status' class='smalltxt' style='color:#b00;display:none;margin-top:6px;'>"
        "Plotly.js unavailable (requires internet connection). Charts not rendered.</div>"
        "</div>"
        # Box plot tab.
        "<div id='prop-tab-box' class='prop-tab-content' style='display:none;'>"
        "<div class='prop-tab-card'>"
        "<div class='smalltxt' style='margin-bottom:8px;'>Use box plots to compare spread, median, interquartile range, and outliers for each property.</div>"
        "<div style='display:flex;flex-wrap:wrap;gap:10px;align-items:center;'>"
        f"<label style='font-size:13px;'>Property: <select id='prop-box-select' onchange='_updatePropBoxPlot()'>{opts_html}</select></label>"
        "<label style='font-size:13px;display:flex;align-items:center;gap:5px;'><input type='checkbox' id='prop-box-filtered' onchange='_updatePropBoxPlot()' /> Pass-filter molecules only</label>"
        "</div>"
        "</div>"
        "<div id='prop-boxplot-chart' style='height:300px;width:100%;min-width:0;'></div>"
        "</div>"
        # Correlation tab.
        "<div id='prop-tab-corr' class='prop-tab-content' style='display:none;'>"
        "<div class='smalltxt' style='margin-bottom:8px;'>"
        "The central map overlays scatter points with a 2D density contour surface. "
        "Use the Plotly toolbar: Box Zoom to focus on a region, Pan to move, "
        "Select/Lasso to mark points, Autoscale or Reset Axes to return."
        "</div>"
        "<div style='display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:10px;'>"
        f"<label style='font-size:13px;'>X axis: <select id='prop-corr-x'>{opts_html}</select></label>"
        f"<label style='font-size:13px;'>Y axis: <select id='prop-corr-y'>{corr_y_opts}</select></label>"
        "<label style='font-size:13px;display:flex;align-items:center;gap:5px;'>"
        "<input type='checkbox' id='prop-corr-filtered' /> Pass-filter molecules only</label>"
        "<button type='button' onclick='_plotCorrelation()' "
        "style='background:#1f3551;color:#fff;border:none;border-radius:6px;padding:5px 14px;cursor:pointer;font-size:13px;'>"
        "Plot</button>"
        "</div>"
        "<div class='prop-corr-wrap' id='prop-corr-wrap'>"
        "<div id='prop-corr-chart' style='height:520px;width:100%;min-width:0;'></div>"
        "<div id='prop-corr-hover' class='prop-corr-hover'></div>"
        "</div>"
        "<div id='prop-corr-click-detail' class='prop-corr-detail empty'>Click a point to pin molecule details.</div>"
        "</div>"
        "</section>"
    )
    return panel


def build_deep_dive_html(
    row,
    subset,
    outdir,
    scaffold_export_data=None,
    top_per_scaffold=12,
    global_reference_smiles=None,
    pose_sdf_by_index=None,
    mol_props_data=None,
    prop_names_list=None,
):
    scaffold_name = str(row.get("scaffold_name", "") or "")
    scaffold_name_js = scaffold_name.replace("\\", "\\\\").replace("'", "\\'")
    checkbox_html = (
        f"<label style='display:inline-flex;align-items:center;gap:5px;cursor:pointer;margin-left:10px;font-weight:normal;'>"
        f"<input type='checkbox' class='scaf-checkbox' data-scaffold='{scaffold_name}' "
        f"onchange=\"syncCheckboxes('{scaffold_name_js}', this.checked)\" />"
        f"<span style='font-size:12px;color:#5f6d7a;'>Select for export</span></label>"
    ) if scaffold_export_data else ""
    deep_dive_id = f"dd-{hash_text(scaffold_name)}"
    scaffold_png = row.get("scaffold_png")
    panel_png = row.get("scaffold_panel_png")
    scaffold_img = _image_data_url(os.path.join(outdir, str(scaffold_png))) if scaffold_png is not None and not pd.isna(scaffold_png) and str(scaffold_png).strip() else ""
    panel_img = _image_data_url(os.path.join(outdir, str(panel_png))) if panel_png is not None and not pd.isna(panel_png) and str(panel_png).strip() else ""

    parts = [f"<div class='card' id='{deep_dive_id}' data-scaffold='{html.escape(scaffold_name, quote=True)}'>"]
    parts.append(
        f"<h3 style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>{row['scaffold_name']}{checkbox_html}"
        f"<button type='button' style='margin-left:auto;background:#fff;border:1px solid #c4d0df;color:#1f3551;border-radius:6px;padding:4px 10px;cursor:pointer;font-size:12px;' onclick=\"clearOverlaySelection('{deep_dive_id}')\">Clear overlays</button>"
        f"</h3>"
    )
    parts.append(
        f"<div class='smalltxt' style='margin-bottom:8px;'><a href='javascript:void(0)' onclick=\"scrollCentralCardIntoView('{scaffold_name_js}')\">Back to central card</a></div>"
    )
    if scaffold_img:
        parts.append(f"<img src='{scaffold_img}' alt='scaffold' decoding='async' style='max-width:500px;width:100%;' />")
    if panel_img:
        parts.append(f"<img src='{panel_img}' alt='scaffold_panel' decoding='async' style='max-width:1100px;width:100%;margin-top:8px;' />")
    parts.append("<ul>")
    parts.append(f"<li><b>Molecules in entry:</b> <span class='dd-members-count' data-raw-members='{int(row['n_members'])}'>{int(row['n_members'])}</span><span class='scaffold-impact-summary smalltxt' data-scaffold='{html.escape(scaffold_name, quote=True)}' style='margin-left:6px;color:#1f4f7a;'></span></li>")
    parts.append(f"<li><b>Median interaction count:</b> {row.get('median_interaction_count')}</li>")
    parts.append(f"<li><b>Median docking score:</b> {row.get('median_score')}</li>")
    parts.append(f"<li><b>Representative IDs:</b> <span class='mono'>{row.get('representative_ids', '')}</span></li>")
    parts.append("<li><b>What to inspect in docking:</b> left-click the molecule tile to open the docking pose viewer.</li>")
    parts.append("</ul>")
    parts.append("<div class='molgrid'>")
    for _, member_row in representative_ranked_subset(subset, top_n=top_per_scaffold).iterrows():
        tile_row = member_row.copy()
        tile_row["search_exact_canonical"] = canonicalize_row_structure(
            member_row,
            pose_sdf_by_index=pose_sdf_by_index,
        )
        parts.append(
            molecule_tile_html(
                tile_row,
                show_scaffold=False,
                global_reference_smiles=global_reference_smiles,
                enable_pose_hover=True,
                enable_overlay_checkbox=True,
                enable_member_checkbox=True,
                member_scaffold_name=scaffold_name,
                member_hbond_residues=extract_hbond_residues_from_row(member_row),
                mol_props_data=mol_props_data,
                prop_names_list=prop_names_list,
            )
        )
    parts.append("</div></div>")
    return "".join(parts)


def _compute_scaffold_prop_ranges(mol_props_data, scaffold_mol_map, prop_names):
    """Compute min/max property ranges for each scaffold.
    Returns: {scaffold_name: {prop: {min, max}}}
    """
    ranges = {}
    if not mol_props_data or not scaffold_mol_map:
        return ranges
    
    for scaf_name, mol_ids in scaffold_mol_map.items():
        ranges[scaf_name] = {}
        for i, prop in enumerate(prop_names):
            vals = []
            for mid in mol_ids:
                if mid in mol_props_data and mol_props_data[mid]:
                    if i < len(mol_props_data[mid]) and mol_props_data[mid][i] is not None:
                        vals.append(mol_props_data[mid][i])
            if vals:
                ranges[scaf_name][prop] = {"min": min(vals), "max": max(vals)}
    return ranges


def _build_mol_smiles_lookup(mol_df):
    lookup = {}
    if mol_df is None or getattr(mol_df, "empty", True):
        return lookup
    for _, row in mol_df.iterrows():
        mol_id = str(row.get("mol_id", "") or "").strip()
        if not mol_id or mol_id in lookup:
            continue
        smiles = str(
            row.get("smiles", "")
            or row.get("exact_canonical_smiles", "")
            or row.get("search_exact_canonical", "")
            or ""
        ).strip()
        if smiles:
            lookup[mol_id] = smiles
    return lookup


def write_html_report(
    outdir,
    mol_df,
    scaf_df,
    central_df,
    qc_df,
    figures,
    scaffold_export_data=None,
    report_filename="report.html",
    min_group_size=3,
    top_per_scaffold=12,
    max_scaffolds_in_report=15,
    global_reference_smiles=None,
    global_reference_core_smarts=None,
    protein_pdb_text=None,
    protein_cartoon_pdb_text=None,
    protein_structure_format="pdb",
    protein_ss_map=None,
    protein_sources=None,
    pose_sdf_by_index=None,
    binding_site_radius=5.0,
    default_pocket_sticks=True,
    hbond_residue_options=None,
    scaffold_hbond_map=None,
    ref_ligand_sdf=None,
    pose_interactions_by_index=None,
    mol_props_data=None,
    scaffold_mol_map=None,
    prop_names_list=None,
    prop_display_labels=None,
):
    html_path = os.path.join(outdir, report_filename)
    structure_assets = ensure_structure_search_assets(outdir)
    if central_df is not None and not central_df.empty and "n_members" in central_df.columns:
        central_df = central_df.loc[
            pd.to_numeric(central_df["n_members"], errors="coerce") >= float(min_group_size)
        ].copy()
    if not global_reference_smiles and scaf_df is not None and not scaf_df.empty:
        from scaffold_summary_helpers import select_depiction_reference_smiles
        global_reference_smiles = select_depiction_reference_smiles(scaf_df)
    report_scaffold_names = central_df.head(max_scaffolds_in_report)["scaffold_name"].tolist() if not central_df.empty else []
    central_page_size = 25
    # mol_df carries scaffold_id but not scaffold_name; merge from scaf_df so
    # build_structure_search_entries can filter and tag entries correctly.
    if "scaffold_name" not in mol_df.columns and scaf_df is not None and not scaf_df.empty and "scaffold_id" in scaf_df.columns and "scaffold_name" in scaf_df.columns:
        _name_map = scaf_df.set_index("scaffold_id")["scaffold_name"].to_dict()
        mol_df = mol_df.copy()
        mol_df["scaffold_name"] = mol_df["scaffold_id"].map(_name_map)
    structure_search_entries_html = build_structure_search_entries(
        mol_df,
        allowed_scaffold_names=report_scaffold_names,
        pose_sdf_by_index=pose_sdf_by_index,
    )
    structure_search_entries_all = build_structure_search_entries(
        mol_df,
        allowed_scaffold_names=None,
        pose_sdf_by_index=pose_sdf_by_index,
    )

    # Parse reference ligands and merge their SDF blocks into the pose index map.
    ref_ligand_entries = parse_ref_ligand_sdf(ref_ligand_sdf) if ref_ligand_sdf else []
    if ref_ligand_entries:
        pose_sdf_by_index = dict(pose_sdf_by_index or {})
        for entry in ref_ligand_entries:
            pose_sdf_by_index[entry["idx"]] = entry["sdf_block"]

    scaffold_unique_member_counts = build_scaffold_unique_member_counts(
        mol_df,
        pose_sdf_by_index=pose_sdf_by_index,
    )

    # Build residue-number → display-name map for H-bond filter labels.
    resi_name_map = build_resi_name_map(protein_pdb_text, protein_structure_format)

    css = """
    :root {
      --ink: #112233;
      --muted: #5f6d7a;
      --accent: #0b6e4f;
      --accent2: #c84c09;
      --bg: #f5f7fb;
      --card: #ffffff;
      --border: #d6dee8;
    }
    body {
      font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
      margin: 0;
      background: linear-gradient(120deg, #eef4fa 0%, #f9fcff 40%, #eefbf4 100%);
      color: var(--ink);
    }
    .wrap { max-width: 1400px; margin: 0 auto; padding: 24px; }
    .hero {
      border: 1px solid var(--border);
      border-radius: 14px;
      background: var(--card);
      padding: 20px 24px;
      margin-bottom: 20px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.05);
    }
    h1 { margin: 0 0 8px 0; font-size: 30px; color: var(--ink); }
    h2 { margin: 18px 0 10px 0; color: var(--ink); }
    h3 { margin: 10px 0; color: #1f3551; }
    p.small { color: var(--muted); margin-top: 0; }
    .kpi {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .kpi .box {
      border: 1px solid var(--border);
      background: #fbfdff;
      border-radius: 10px;
      padding: 10px;
    }
    .kpi .label { color: var(--muted); font-size: 12px; }
    .kpi .val { font-size: 20px; font-weight: 700; color: var(--accent); }
    .tbl {
      border-collapse: collapse;
      width: 100%;
      margin-bottom: 16px;
      background: var(--card);
    }
    .tbl th, .tbl td {
      border: 1px solid var(--border);
      padding: 7px;
      vertical-align: top;
      font-size: 12px;
    }
    .tbl th { background: #edf3f9; }
    .panel {
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--card);
      padding: 14px;
      margin: 12px 0;
    }
    .panel.collapsed > *:not(h2) { display: none; }
    .panel.collapsible > h2 { cursor: pointer; user-select: none; }
    .panel.collapsible > h2::after { content: ' \25BC'; font-size: 10px; color: #888; margin-left: 6px; }
    .panel.collapsible.collapsed > h2::after { content: ' \25B6'; }
    .plot-grid {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
    }
    .plot-grid img { width: 100%; border-radius: 8px; border: 1px solid var(--border); }
    .card {
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      margin-top: 12px;
      background: #ffffff;
    }
    .smalltxt { color: var(--muted); font-size: 12px; }
    .mono { font-family: Consolas, Menlo, monospace; font-size: 11px; }
    .molgrid {
      display: grid;
      gap: 8px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }
    .moltile {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px;
      background: #fcfdff;
    }
    .moltile img { max-width: 100%; height: auto; }
    .prop-tab-bar { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:12px; border-bottom:2px solid var(--border); padding-bottom:4px; }
    .prop-tab-btn { background:#f0f4f8; border:1px solid #cdd8e6; border-radius:6px 6px 0 0; padding:6px 14px; cursor:pointer; font-size:13px; color:#334; }
    .prop-tab-btn.active { background:#fff; border-bottom-color:#fff; color:#0b6e4f; font-weight:600; }
    .prop-tab-content { padding:12px 0; }
    .prop-tab-card { border:1px solid #d7e5f3; border-radius:10px; background:linear-gradient(180deg,#fbfdff 0%,#f6fbff 100%); padding:10px 12px; margin-bottom:10px; }
    .prop-stats-table { border-collapse:collapse; width:100%; font-size:13px; }
    .prop-stats-table th { background:#f0f4f8; text-align:left; padding:6px 10px; border:1px solid #cdd8e6; color:#1f3551; font-weight:600; }
    .prop-stats-table td { padding:5px 10px; border:1px solid #dde4ee; }
    .prop-stats-table tr:nth-child(even) td { background:#fafbfd; }
    .prop-filter-chip-tag { display:inline-flex; align-items:center; gap:4px; background:#ddeeff; border:1px solid #8bbfe8; border-radius:999px; padding:2px 8px; margin:2px; font-size:12px; }
    .prop-filter-chip-tag button { background:none; border:none; cursor:pointer; color:#555; font-size:14px; line-height:1; padding:0 0 0 2px; }
    #prop-active-filters { background:#fff8e1; border:1px solid #ffe082; border-radius:6px; padding:8px 12px; margin-bottom:10px; display:flex; flex-wrap:wrap; align-items:center; gap:8px; }
    .prop-hist-controls { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-top:8px; }
    .prop-range-field { display:flex; align-items:center; gap:4px; font-size:13px; }
    .prop-range-field input { width:90px; padding:4px 6px; border:1px solid #cdd; border-radius:4px; }
        .idea-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 12px;
        }
        .idea-card {
            border: 1px solid var(--border);
            border-radius: 12px;
            background: #fbfdff;
            padding: 12px;
        }
        .idea-head {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 8px;
        }
        .central-filter-impact {
            margin: 10px 0 14px;
            padding: 10px 12px;
            border: 1px solid #d7e5f3;
            border-radius: 10px;
            background: #f7fbff;
        }
        .central-filter-impact-summary {
            font-weight: 600;
            color: #1f3551;
        }
        .central-filter-impact-breakdown {
            margin-top: 6px;
            color: #46607a;
        }
        .prop-corr-wrap {
            position: relative;
        }
        .prop-corr-hover {
            position: fixed;
            z-index: 1200;
            display: none;
            width: 240px;
            max-width: min(240px, calc(100vw - 24px));
            background: rgba(255, 255, 255, 0.97);
            border: 1px solid #cad8e6;
            border-radius: 10px;
            box-shadow: 0 10px 28px rgba(31, 53, 81, 0.18);
            padding: 8px;
            pointer-events: none;
        }
        .prop-corr-structure {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 140px;
            background: #fff;
            border: 1px solid #e2ebf3;
            border-radius: 8px;
            overflow: hidden;
        }
        .prop-corr-structure svg {
            width: 100%;
            height: auto;
            display: block;
        }
        .prop-corr-meta {
            margin-top: 8px;
            color: #23384f;
            font-size: 12px;
            line-height: 1.45;
        }
        .prop-corr-detail {
            margin-top: 10px;
            padding: 10px 12px;
            border: 1px solid #d7e5f3;
            border-radius: 10px;
            background: #f8fbfe;
            min-height: 42px;
        }
        .prop-corr-detail.empty {
            color: #6a7a89;
        }
        .export-toolbar {
            position: sticky;
            top: 0;
            z-index: 200;
            background: var(--accent);
            color: #fff;
            padding: 10px 20px;
            display: none;
            align-items: center;
            gap: 14px;
            border-radius: 0 0 10px 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            font-weight: 600;
            font-size: 14px;
        }
        .export-toolbar button {
            background: #fff;
            color: var(--accent);
            border: none;
            border-radius: 6px;
            padding: 6px 16px;
            cursor: pointer;
            font-weight: 600;
            font-size: 13px;
        }
        .export-toolbar button:hover { background: #e0f0ea; }
        .export-toolbar .clr-btn {
            background: transparent;
            color: #cde;
            border: 1px solid rgba(255, 255, 255, 0.5);
        }
        .export-toolbar .clr-btn:hover { background: rgba(255, 255, 255, 0.15); }
        .export-group { display:none; align-items:center; gap:14px; }
        .export-group + .export-group { padding-left:14px; border-left:1px solid rgba(255,255,255,0.35); }
        .scaf-checkbox { width: 16px; height: 16px; cursor: pointer; accent-color: var(--accent); }
        .member-export-toggle { width: 16px; height: 16px; cursor: pointer; accent-color: #1f3551; }
        .idea-card.sel-active { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(11, 110, 79, 0.3); }
        .idea-card.is-deactivated { opacity: 0.42; filter: grayscale(0.35); background: #f6f7f9; }
        .idea-card.is-deactivated a,
        .idea-card.is-deactivated button,
        .idea-card.is-deactivated input,
        .idea-card.is-deactivated select,
        .idea-card.is-deactivated textarea,
        .idea-card.is-deactivated .moltile { pointer-events: none; }
        .idea-card.is-deactivated .deactivate-control,
        .idea-card.is-deactivated .deactivate-control * { pointer-events: auto; }
        .card.sel-active { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(11, 110, 79, 0.3); }
        .moltile.member-sel-active { border-color: #1f3551 !important; box-shadow: 0 0 0 2px rgba(31, 53, 81, 0.2); background:#f7fbff; }
        .star-wrap { display:inline-flex; align-items:center; gap:4px; margin-left:6px; }
        .star-btn { border: 1px solid #c4d0df; background: #fff; color: #223; border-radius: 4px; padding: 2px 6px; cursor: pointer; font-size: 11px; }
        .star-btn:hover { background: #f2f6fb; }
        .star-btn.active { border-color: #c84c09; background: #fff2e8; color: #8b2f00; font-weight: 700; }
        .star-btn.star-clear { color: #6a7280; }
        .deactivate-control { color: #5f6d7a; font-weight: 600; }
        .deactivate-control input { accent-color: #8a1f1f; cursor: pointer; }
        .panel-actions { display:flex; align-items:center; gap:8px; margin:6px 0 10px 0; }
        .panel-actions button { background:#fff; border:1px solid #c4d0df; color:#1f3551; border-radius:6px; padding:5px 10px; cursor:pointer; font-size:12px; }
        .panel-actions input[type='search'] { background:#fff; border:1px solid #c4d0df; color:#1f3551; border-radius:6px; padding:5px 10px; font-size:12px; min-width:230px; }
        .panel-actions .filter-chip { display:inline-flex; align-items:center; gap:5px; background:#fff; border:1px solid #c4d0df; color:#1f3551; border-radius:6px; padding:4px 8px; font-size:12px; }
        .panel-actions button:hover { background:#f2f6fb; }
        .pager-bar { display:flex; align-items:center; gap:10px; justify-content:flex-end; margin:0 0 10px 0; flex-wrap:wrap; }
        .pager-bar button { background:#fff; border:1px solid #c4d0df; color:#1f3551; border-radius:6px; padding:5px 12px; cursor:pointer; font-size:12px; }
        .pager-bar button:disabled { cursor:not-allowed; opacity:0.45; }
        .pager-status { color:#5f6d7a; font-size:12px; min-width:120px; text-align:center; }
        .pager-select { background:#fff; border:1px solid #c4d0df; color:#1f3551; border-radius:6px; padding:5px 8px; font-size:12px; }
        .highlight-list { margin:8px 0 10px 0; padding:8px 10px; border:1px solid #d6dee8; border-radius:8px; background:#fbfdff; font-size:12px; color:#33475e; }
        .highlight-list .tag { display:inline-block; margin:2px 6px 2px 0; padding:2px 8px; border-radius:999px; font-weight:600; }
        .highlight-list .tag.red { background:#fff2e8; color:#8b2f00; border:1px solid #f1c7ad; }
        .highlight-list .tag.green { background:#eefaf1; color:#1d6134; border:1px solid #bfe3cb; }
        .lazy-placeholder { border:1px dashed #c4d0df; border-radius:12px; padding:16px; background:#fbfdff; color:#5f6d7a; }
        body.pose-popup-mode { padding: 0; background: #eef3f8; }
        body.pose-popup-mode > * { display: none !important; }
        body.pose-popup-mode #pose-panel { display: flex !important; position: fixed !important; inset: 0 !important; width: 100vw !important; height: 100vh !important; min-width: 0 !important; min-height: 0 !important; max-width: none !important; max-height: none !important; right: auto !important; bottom: auto !important; border: none !important; border-radius: 0 !important; box-shadow: none !important; }
        body.pose-popup-mode #pose-close { display: none !important; }
        html.pose-popup-early { background: #0f1923; }
        html.pose-popup-early body > * { display: none !important; }
        .residue-filter-grid { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
        .residue-filter-chip { display:inline-flex; align-items:center; gap:6px; background:#f8fbff; border:1px solid #cfe0ef; border-radius:999px; padding:6px 10px; font-size:12px; color:#1f3551; }
        .residue-filter-chip input { accent-color: var(--accent); }
        .structure-search-panel details { border: 1px solid #cfe0ef; border-radius: 12px; background: linear-gradient(180deg, #f8fbff 0%, #fcfffd 100%); }
        .structure-search-panel summary { list-style: none; cursor: pointer; padding: 14px 16px; display:flex; align-items:center; justify-content:space-between; gap:12px; font-weight:700; color:#15324f; }
        .structure-search-panel summary::-webkit-details-marker { display:none; }
        .structure-search-panel summary .smalltxt { margin-left:auto; }
        .structure-search-body { padding: 0 16px 16px 16px; display:flex; flex-direction:column; gap:10px; max-width:980px; }
        .structure-search-body textarea { min-height:110px; resize:vertical; border:1px solid #c4d0df; border-radius:8px; padding:10px 12px; font: inherit; color:#1f3551; background:#fff; width:100%; box-sizing:border-box; }
        .structure-search-mode { display:flex; flex-wrap:wrap; gap:10px; }
        .structure-search-mode label { display:inline-flex; align-items:center; gap:6px; background:#fff; border:1px solid #c4d0df; border-radius:999px; padding:6px 12px; font-size:12px; color:#1f3551; }
        .structure-search-mode input { accent-color: var(--accent); }
        .search-scope-tabs { display:inline-flex; gap:6px; margin-bottom:8px; padding:4px; border:1px solid #cfd9e4; border-radius:10px; background:#f4f8fc; }
        .search-scope-tab { border:1px solid #c4d0df; border-radius:8px; padding:5px 10px; background:#ffffff; color:#2c4056; font-size:12px; cursor:pointer; }
        .search-scope-tab.active { background:#0b6e4f; border-color:#0b6e4f; color:#ffffff; font-weight:600; }
        .structure-search-buttons { display:flex; flex-wrap:wrap; gap:8px; }
        .structure-search-buttons button { background:#fff; border:1px solid #c4d0df; color:#1f3551; border-radius:8px; padding:7px 12px; cursor:pointer; font-size:12px; font-weight:600; }
        .structure-search-buttons button.primary { background:#0b6e4f; border-color:#0b6e4f; color:#fff; }
        .structure-search-buttons button:hover { background:#f2f6fb; }
        .structure-search-buttons button.primary:hover { background:#0a5f45; }
        .structure-search-status { border-radius:8px; padding:10px 12px; font-size:12px; border:1px solid #cfe0ef; background:#fff; color:#1f3551; }
        .structure-search-status[data-tone='ready'] { border-color:#bfdcf5; background:#eef8ff; color:#1f4f7a; }
        .structure-search-status[data-tone='working'] { border-color:#f8d88a; background:#fff9e8; color:#7a5b00; }
        .structure-search-status[data-tone='error'] { border-color:#f3c7c7; background:#fff1f1; color:#8a1f1f; }
        .structure-search-status[data-tone='success'] { border-color:#bfe3cb; background:#eefaf1; color:#1d6134; }
        .search-status-line { display:block; }
        .search-progress-track { display:none; margin-top:8px; height:6px; border-radius:999px; background:#dce7f3; overflow:hidden; }
        .search-progress-fill { width:0%; height:100%; border-radius:999px; background:linear-gradient(90deg,#0b6e4f,#1c8f69); transition:width 120ms linear; }
        .search-scope-badge { display:inline-flex; align-items:center; margin-left:8px; padding:1px 6px; border:1px solid #9dc2de; border-radius:999px; background:#ffffff; color:#1f4f7a; font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.02em; }
        .idea-card.structure-match-active, .card.structure-match-active, .moltile.structure-match-active { border-color:#1f4f7a !important; box-shadow:0 0 0 2px rgba(31,79,122,0.18); }
    """

    kpi = {
        "Molecules": int(len(mol_df)),
        "Molecule List Entries": int(len(central_df)),
    }
    qc_map = dict(zip(qc_df["metric"], qc_df["value"])) if not qc_df.empty else {}
    report_hbd_violations = int(float(qc_map.get("report_hbd_violations", 0) or 0))

    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write("<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>")
        fh.write(f"<style>{css}</style>")
        fh.write("<script>(function(){try{if((new URLSearchParams(window.location.search)).get('posePopup')==='1'){document.documentElement.classList.add('pose-popup-early');}}catch(e){}})()</script>")
        fh.write("</head><body>")
        if scaffold_export_data:
            fh.write("<div id='export-toolbar' class='export-toolbar'>")
            fh.write("<div id='scaf-export-group' class='export-group'>")
            fh.write("<span>&#9745;&nbsp;<span id='sel-count'>0</span>&nbsp;scaffold(s) selected</span>")
            fh.write("<button onclick='exportCSV()'>&#11015; Export CSV (all members)</button>")
            fh.write("<button onclick='exportSDF()'>&#11015; Export Poses SDF (3D)</button>")
            fh.write("<button class='clr-btn' onclick='clearSel()'>&#10005; Clear scaffolds</button>")
            fh.write("</div>")
            fh.write("<div id='member-export-group' class='export-group'>")
            fh.write("<span>&#9745;&nbsp;<span id='member-sel-count'>0</span>&nbsp;member(s) selected</span>")
            fh.write("<button onclick='exportMemberSDF()'>&#11015; Export Selected Members SDF</button>")
            fh.write("<button class='clr-btn' onclick='clearMemberSel()'>&#10005; Clear members</button>")
            fh.write("</div>")
            fh.write("</div>")
        fh.write("<div class='wrap'>")

        fh.write("<section class='hero'>")
        fh.write("<h1>Docking Molecule Insight Report</h1>")
        fh.write("<p class='small'>Molecule-level prioritization from docking SDF plus interaction-count CSV.</p>")
        fh.write("<div class='kpi'>")
        for key, val in kpi.items():
            fh.write(f"<div class='box'><div class='label'>{key}</div><div class='val'>{val}</div></div>")
        fh.write("</div></section>")

        fh.write("<section class='panel collapsible'><h2 ondblclick=\"this.closest('.panel').classList.toggle('collapsed')\">Help: How to Navigate This Report</h2>")
        fh.write(
            "<p class='smalltxt'>"
            "This user guide explains how to navigate and utilize an interactive HTML docking report for exploring molecular docking results. "
            "It covers workflows for data exploration, molecule downloading, 3D visualization, filtering, and troubleshooting. "
            "<b>Tip:</b> double-click the section title to collapse or expand this panel."
            "</p>"
        )
        fh.write("<ul>")
        fh.write(
            "<li><b>Report structure and navigation:</b> "
            "The report contains three main sections: Overview, Molecule List, and deep dive details for each molecule. "
            "Users can navigate from summary molecule cards to detailed views showing 3D structures, docking scores, and interaction fingerprints.</li>"
        )
        fh.write(
            "<li><b>Structure search functionality:</b> "
            "Users can filter molecules by chemical structure using SMILES or SMARTS queries in substructure or exact match modes, with examples and reset options provided.</li>"
        )
        fh.write(
            "<li><b>Exclusion motif filtering:</b> "
            "The Exclude Motif panel removes matching molecules globally from the report list and deep-dive tiles. "
            "Multiple exclusion motifs can be active at once, can be scoped to HTML subset or all report-eligible molecules, and can be cleared with Reset Exclusion.</li>"
        )
        fh.write(
            "<li><b>Hydrogen-bond residue filtering:</b> "
            "The report allows filtering molecules based on contacting specific protein residues using AND logic.</li>"
        )
        fh.write(
            "<li><b>Downloading molecules:</b> "
            "Selected molecules can be inspected in deep dive cards and docking view.</li>"
        )
        fh.write(
            "<li><b>Docking pose visualizer:</b> "
            "Each deep dive includes a 3D viewer displaying the protein, docked ligand, and binding pocket, with controls for rotation, zoom, and panning.</li>"
        )
        fh.write(
            "<li><b>Overlay feature:</b> "
            "Users can overlay multiple molecules (2-5) in the 3D viewer to compare docking poses and interactions.</li>"
        )
        fh.write("</ul>")
        fh.write(
            "<p class='smalltxt'>"
            "Detailed tutorial: "
            "<a href='https://gileadconnect-my.sharepoint.com/:w:/g/personal/panchamlal_gupta11_gilead_com/IQCIyWedvq7uQb-uLbLVrPOzAaRtFZA6uKO5KOxErvR4AcM?e=ahW8cd' target='_blank' rel='noopener noreferrer'>"
            "Open full guide"
            "</a>"
            "</p>"
        )
        fh.write("</section>")

        if hbond_residue_options:
            fh.write("<section class='panel'><h2>Hydrogen Bonding Residues</h2>")
            fh.write("<p class='smalltxt'>Select one or more residues to keep only molecules making all selected hydrogen bonds, based on donor/acceptor interaction columns in the interaction CSV.</p>")
            fh.write("<div class='residue-filter-grid'>")
            for residue in hbond_residue_options:
                # value stays as raw number so JS filtering is unaffected
                aa_name = resi_name_map.get(str(residue), "")
                display_label = f"{aa_name}{residue}" if aa_name else str(residue)
                fh.write(
                    "<label class='residue-filter-chip'>"
                    f"<input type='checkbox' class='hbond-residue-filter' value='{residue}' />"
                    f"<span>{html.escape(display_label)}</span>"
                    "</label>"
                )
            fh.write("</div>")
            fh.write("</section>")

        fh.write("<section class='panel structure-search-panel'><details open>")
        fh.write("<summary><span>Structure Search</span><span class='smalltxt'>Paste SMILES or SMARTS, then press Search</span></summary>")
        fh.write("<div class='structure-search-body'>")
        fh.write("<p class='smalltxt'>Paste a SMILES or SMARTS string to filter molecule cards and deep-dive sections by substructure or exact full-molecule match.</p>")
        fh.write("<div class='search-scope-tabs' role='tablist' aria-label='Structure search scope'>")
        fh.write("<button type='button' class='search-scope-tab active' data-search-scope='html_subset' role='tab' aria-selected='true'>HTML page molecules subset</button>")
        fh.write("<button type='button' class='search-scope-tab' data-search-scope='all_report_eligible' role='tab' aria-selected='false'>All report-eligible molecules</button>")
        fh.write("</div>")
        fh.write("<textarea id='structure-search-input' placeholder='Paste SMILES or SMARTS here (e.g. c1ccccc1 or [#6]-[#7])'></textarea>")
        fh.write("<div class='structure-search-mode'>")
        fh.write("<label><input type='radio' name='structure-search-mode' value='substructure' checked /> Substructure search</label>")
        fh.write("<label><input type='radio' name='structure-search-mode' value='exact' /> Exact molecule match</label>")
        fh.write("</div>")
        fh.write("<div class='structure-search-buttons'>")
        fh.write("<button type='button' class='primary' id='structure-search-run'>Search</button>")
        fh.write("<button type='button' id='structure-search-reset'>Reset Search</button>")
        fh.write("</div>")
        fh.write("<div class='structure-search-status' id='structure-search-status' data-tone='ready'><span class='search-status-line'>Molecule list loads first. RDKit search loads on first use.</span><span class='search-scope-badge' id='structure-search-scope-badge'></span><div class='search-progress-track'><div class='search-progress-fill' id='structure-search-progress-fill'></div></div></div>")
        fh.write("</div></details></section>")

        fh.write("<section class='panel structure-search-panel'><details open>")
        fh.write("<summary><span>Exclude Motif</span><span class='smalltxt'>Paste SMILES or SMARTS to remove matching molecules</span></summary>")
        fh.write("<div class='structure-search-body'>")
        fh.write("<p class='smalltxt'>Matches are removed globally from molecule counts and visible deep-dive tiles. Pasted SMILES are treated as substructure queries by default.</p>")
        fh.write("<div class='search-scope-tabs' role='tablist' aria-label='Exclusion motif scope'>")
        fh.write("<button type='button' class='search-scope-tab active' data-search-scope='html_subset' role='tab' aria-selected='true'>HTML page molecules subset</button>")
        fh.write("<button type='button' class='search-scope-tab' data-search-scope='all_report_eligible' role='tab' aria-selected='false'>All report-eligible molecules</button>")
        fh.write("</div>")
        fh.write("<textarea id='exclude-motif-input' placeholder='Paste SMILES or SMARTS to exclude (e.g. c1ccccc1 or [#6]-[#7])'></textarea>")
        fh.write("<div class='structure-search-buttons'>")
        fh.write("<button type='button' class='primary' id='exclude-motif-run'>Apply Exclusion</button>")
        fh.write("<button type='button' id='exclude-motif-reset'>Reset Exclusion</button>")
        fh.write("</div>")
        fh.write("<div class='structure-search-status' id='exclude-motif-status' data-tone='ready'><span class='search-status-line'>No motif exclusion is active.</span><div class='search-progress-track'><div class='search-progress-fill' id='exclude-motif-progress-fill'></div></div></div>")
        fh.write("<div id='exclude-motif-active-list' style='display:flex;flex-direction:column;gap:8px;'></div>")
        fh.write("</div></details></section>")

        # Properties Panel (above Molecule List).
        _prop_names_ordered = list(prop_names_list or [
            "GS_LogD", "GS_Sol_74_linear", "GS_CACO2_A2B_10_linear", "GS_CACO2_B2A_10_linear",
            "GS_HP_Free_LT_linear", "GS_CACO2_A2B_1_linear", "GS_CACO2_B2A_1_linear",
            "GS_HP_Free_linear", "GS_Pred_Cl_HLM_linear", "GS_MDCK_linear", "GS_RED_HP_linear",
            "interaction_count",
            "MW", "cLogP", "TPSA", "HBD", "HBA", "RotBonds", "HeavyAtoms", "FormalCharge",
            "RingCount", "FractionCSP3",
        ])
        _has_props = bool(mol_props_data)
        if _has_props:
            _prop_stats = _compute_prop_stats(mol_props_data, _prop_names_ordered)
            _mol_smiles_lookup = _build_mol_smiles_lookup(mol_df)
            fh.write(_build_prop_panel_html(_prop_names_ordered, _prop_stats, prop_display_labels))
            # Serialize data constants for JS.
            _effective_prop_labels = dict(_PROP_DISPLAY_LABELS)
            _effective_prop_labels.update(prop_display_labels or {})
            _prop_labels_js = {p: _effective_prop_labels.get(p, p) for p in _prop_names_ordered}
            _prop_names_js = json.dumps(_prop_names_ordered, ensure_ascii=True)
            _prop_labels_json = json.dumps(_prop_labels_js, ensure_ascii=True)
            _mol_props_json = json.dumps(mol_props_data, ensure_ascii=True)
            _scaffold_mol_json = json.dumps(scaffold_mol_map or {}, ensure_ascii=True)
            _mol_props_json_str = json.dumps(_mol_props_json, ensure_ascii=True)
            _scaffold_mol_json_str = json.dumps(_scaffold_mol_json, ensure_ascii=True)
            _mol_smiles_json = json.dumps(_mol_smiles_lookup, ensure_ascii=True)
            fh.write(
                f"<script>"
                f"const _PROP_NAMES={_prop_names_js};"
                f"const _PROP_LABELS={_prop_labels_json};"
                f"const _MOL_PROPS_DATA_JSON={_mol_props_json_str};"
                f"const _SCAFFOLD_MOL_MAP_JSON={_scaffold_mol_json_str};"
                f"var _MOL_PROPS_DATA=null;"
                f"var _SCAFFOLD_MOL_MAP=null;"
                f"const _MOL_SMILES_BY_ID={_mol_smiles_json};"
                f"var _PROP_FILTER_STATE={{}};"
                f"</script>"
            )

        fh.write("<section class='panel'><h2>Molecule List</h2>")
        fh.write(
            "<p class='smalltxt'>All report molecules from the input SDF are listed here. "
            "Click a molecule image to open the docking pose visualizer.</p>"
        )
        if report_hbd_violations > 0:
            fh.write(
                f"<p class='smalltxt' style='color:#b42318;'><b>Warning:</b> {report_hbd_violations} report molecules exceed current HBD limit and should be inspected.</p>"
            )
        else:
            fh.write("<p class='smalltxt' style='color:#0b6e4f;'><b>HBD gate check:</b> all report molecules satisfy the configured HBD limit.</p>")
        fh.write(
            "<div class='panel-actions'>"
            "<button type='button' onclick='sortStarredToTop()'>Sort Starred to Top</button>"
            "<button type='button' onclick='resetScaffoldOrder()'>Reset List</button>"
            "<button type='button' onclick='clearAllStars()'>Clear All Stars</button>"
            "<button type='button' onclick='activateAllScaffolds()'>Activate All</button>"
            "<input type='search' id='central-scaffold-search' placeholder='Search molecule ID' />"
            "<label class='filter-chip'><input type='checkbox' id='filter-hide-duplicates' checked /> Hide duplicate molecules (canonical SMILES)</label>"
            "</div>"
        )
        top15_names = (
            central_df.sort_values(["n_members", "central_priority"], ascending=[False, False]).head(15)["scaffold_name"].tolist()
            if not central_df.empty else []
        )
        central_card_payload = build_central_card_payload(
            central_df,
            highlighted_scaffold_names=top15_names,
            scaffold_export_data=scaffold_export_data,
            scaffold_hbond_map=scaffold_hbond_map,
            top_per_scaffold=top_per_scaffold,
            mol_props_data=mol_props_data,
            scaffold_mol_map=scaffold_mol_map,
            prop_names_ordered=_prop_names_ordered,
            scaffold_unique_member_counts=scaffold_unique_member_counts,
            global_reference_smiles=global_reference_smiles,
            global_reference_core_smarts=global_reference_core_smarts,
        )
        report_render_payload = {
            "pageSize": central_page_size,
            "centralCards": central_card_payload,
        }
        report_render_payload_json = json.dumps(report_render_payload, ensure_ascii=True)
        report_render_payload_script = f"<script>const _REPORT_RENDER_PAYLOAD={report_render_payload_json};</script>"
        fh.write(
            "<div class='pager-bar'>"
            "<label class='smalltxt' for='central-page-size' style='font-weight:600;'>Per page</label>"
            "<select id='central-page-size' class='pager-select'>"
            "<option value='25' selected>25</option>"
            "<option value='50'>50</option>"
            "<option value='100'>100</option>"
            "<option value='200'>200</option>"
            "<option value='500'>500</option>"
            "<option value='all'>All</option>"
            "</select>"
            "<button type='button' id='central-page-first'>&laquo; First</button>"
            "<button type='button' id='central-page-prev'>&larr; Prev</button>"
            "<span class='pager-status' id='central-page-status'>Page 1 of 1</span>"
            "<button type='button' id='central-page-next'>Next &rarr;</button>"
            "<button type='button' id='central-page-last'>Last &raquo;</button>"
            "</div>"
        )
        fh.write("<div class='idea-grid' id='central-idea-grid'></div>")
        fh.write(
            "<div class='central-filter-impact' id='central-filter-impact' style='display:none;'>"
            "<div class='central-filter-impact-summary smalltxt' id='central-filter-impact-summary'></div>"
            "<div class='central-filter-impact-breakdown smalltxt' id='central-filter-impact-breakdown'></div>"
            "</div>"
        )
        fh.write(
            "<div class='pager-bar'>"
            "<button type='button' id='central-page-first-bottom'>&laquo; First</button>"
            "<button type='button' id='central-page-prev-bottom'>&larr; Prev</button>"
            "<span class='pager-status' id='central-page-status-bottom'>Page 1 of 1</span>"
            "<button type='button' id='central-page-next-bottom'>Next &rarr;</button>"
            "<button type='button' id='central-page-last-bottom'>Last &raquo;</button>"
            "</div>"
        )
        fh.write("</section>")

        export_json = json.dumps(scaffold_export_data or {}, ensure_ascii=True)
        protein_json = json.dumps(protein_pdb_text or "", ensure_ascii=True)
        protein_cartoon_json = json.dumps(protein_cartoon_pdb_text or "", ensure_ascii=True)
        protein_fmt_str = (protein_structure_format or "pdb").strip().lower()
        protein_fmt_json = json.dumps(protein_fmt_str, ensure_ascii=True)
        protein_ss_json = json.dumps(protein_ss_map or {}, ensure_ascii=True)
        protein_sources_payload = []
        for entry in protein_sources or []:
            chem_text = str(entry.get("chem_text") or "")
            cartoon_text = str(entry.get("cartoon_text") or "")
            chem_format = str(entry.get("chem_format") or "pdb").strip().lower()
            if chem_format == "mol2" and chem_text:
                serial_map = parse_mol2_serial_map(chem_text)
            elif cartoon_text:
                serial_map = parse_pdb_serial_map(cartoon_text)
            elif chem_text:
                serial_map = parse_pdb_serial_map(chem_text)
            else:
                serial_map = {}
            protein_sources_payload.append(
                {
                    "id": str(entry.get("id") or f"protein-{len(protein_sources_payload) + 1}"),
                    "label": str(entry.get("label") or f"Protein {len(protein_sources_payload) + 1}"),
                    "chemText": chem_text,
                    "chemFormat": chem_format,
                    "cartoonText": cartoon_text,
                    "serialMap": serial_map,
                    "ssMap": entry.get("ss_map") or {},
                }
            )
        protein_sources_json = json.dumps(protein_sources_payload, ensure_ascii=True)
        # Build serial→residue map for reliable hover labels regardless of parser behaviour
        if protein_fmt_str == "mol2" and protein_pdb_text:
            protein_serial_map = parse_mol2_serial_map(protein_pdb_text)
        elif protein_cartoon_pdb_text:
            protein_serial_map = parse_pdb_serial_map(protein_cartoon_pdb_text)
        elif protein_pdb_text:
            protein_serial_map = parse_pdb_serial_map(protein_pdb_text)
        else:
            protein_serial_map = {}
        protein_serial_map_json = json.dumps(protein_serial_map, ensure_ascii=True)
        pose_json = json.dumps({str(key): value for key, value in (pose_sdf_by_index or {}).items()}, ensure_ascii=True)
        pose_interactions_json = json.dumps(
            {str(key): value for key, value in (pose_interactions_by_index or {}).items()},
            ensure_ascii=True,
        )
        ref_ligand_options_json = json.dumps(
            [
                {"idx": int(entry.get("idx", -1)), "label": str(entry.get("title", ""))}
                for entry in (ref_ligand_entries or [])
                if entry.get("idx") is not None
            ],
            ensure_ascii=True,
        )
        hbond_json = json.dumps(scaffold_hbond_map or {}, ensure_ascii=True)
        structure_search_html_json = json.dumps(structure_search_entries_html, ensure_ascii=True)
        structure_search_all_json = json.dumps(structure_search_entries_all, ensure_ascii=True)
        # Keep embedded wasm for file:// openings (browser CORS blocks local wasm fetches).
        rdkit_wasm_b64_json = json.dumps(get_embedded_rdkit_wasm_b64(), ensure_ascii=True)
        binding_radius = float(binding_site_radius) if binding_site_radius is not None else 5.0
        default_pocket_sticks_js = "true" if default_pocket_sticks else "false"
        js_block = build_docking_pose_visualizer_js(
            exp_json=export_json,
            has_export=bool(scaffold_export_data),
            protein_json=protein_json,
            protein_cartoon_json=protein_cartoon_json,
            protein_fmt_json=protein_fmt_json,
            protein_ss_json=protein_ss_json,
            protein_serial_map_json=protein_serial_map_json,
            protein_sources_json=protein_sources_json,
            pose_json=pose_json,
            pose_interactions_json=pose_interactions_json,
            ref_ligand_options_json=ref_ligand_options_json,
            binding_radius=binding_radius,
            default_pocket_sticks_js=default_pocket_sticks_js,
        )
        fh.write(report_render_payload_script)
        fh.write(f"<script src='{structure_assets['rdkit_js_src']}'></script>")
        fh.write(js_block)
        fh.write(
            "<script>\n"
            f"const _HBOND_BY_SCAFFOLD={hbond_json};\n"
            f"const _STRUCTURE_SEARCH_ENTRIES_HTML={structure_search_html_json};\n"
            f"const _STRUCTURE_SEARCH_ENTRIES_ALL={structure_search_all_json};\n"
            "const _DEFAULT_SEARCH_SCOPE='html_subset';\n"
            "const _STRUCTURE_MATCH_STATE={active:false,mode:'substructure',scope:_DEFAULT_SEARCH_SCOPE,matchedScaffolds:null,matchedMolIds:null,lastQuery:''};\n"
            "const _EXCLUDE_MOTIF_STATE={items:[],nextId:1,lastQuery:''};\n"
            f"const _RDKIT_WASM_PATH={json.dumps(structure_assets['rdkit_wasm_src'], ensure_ascii=True)};\n"
            f"const _RDKIT_WASM_B64={rdkit_wasm_b64_json};\n"
            "let _rdkitReadyPromise=null;\n"
            "let _structureLibraryByScope={};\n"
            "let _plotlyReadyPromise=null;\n"
            "let _structureWarmupStarted=false;\n"
            "const _DEACTIVATE_KEY='rgroup_report_deactivated_v1';\n"
            "const _DEACTIVATED={};\n"
            "try{var _savedDeactivated=localStorage.getItem(_DEACTIVATE_KEY);if(_savedDeactivated){Object.assign(_DEACTIVATED,JSON.parse(_savedDeactivated));}}catch(_e){}\n"
            "const _CENTRAL_RENDER_STATE={page:0,pageSize:25,sortMode:'initial',filteredScaffolds:[],activeDeepDive:'',lastTotalPages:1,scaffoldPropStats:{},propImpactSummary:null};\n"
            "const _POSE_POPUP_QUERY_KEY='posePopup';\n"
            "var _POSE_POPUP_REF=null;\n"
            "function _isScaffoldDeactivated(name){return !!_DEACTIVATED[String(name||'')];}\n"
            "function _persistDeactivatedScaffolds(){try{localStorage.setItem(_DEACTIVATE_KEY,JSON.stringify(_DEACTIVATED));}catch(_e){}}\n"
            "function setScaffoldDeactivated(name,deactivated){\n"
            "  var key=String(name||'');\n"
            "  if(!key){return;}\n"
            "  if(deactivated){_DEACTIVATED[key]=true;}else{delete _DEACTIVATED[key];}\n"
            "  _persistDeactivatedScaffolds();\n"
            "  if(_CENTRAL_RENDER_STATE.activeDeepDive===key&&_isScaffoldDeactivated(key)){_CENTRAL_RENDER_STATE.activeDeepDive='';}\n"
            "  _renderCentralPage();\n"
            "}\n"
            "function activateAllScaffolds(){\n"
            "  Object.keys(_DEACTIVATED).forEach(function(key){delete _DEACTIVATED[key];});\n"
            "  _persistDeactivatedScaffolds();\n"
            "  _renderCentralPage();\n"
            "}\n"
            "function _syncDeactivatedScaffolds(){\n"
            "  document.querySelectorAll('.deactivate-toggle').forEach(function(cb){\n"
            "    var scaf=String(cb.dataset.scaffold||'');\n"
            "    cb.checked=_isScaffoldDeactivated(scaf);\n"
            "  });\n"
            "  document.querySelectorAll('#central-idea-grid .idea-card').forEach(function(card){\n"
            "    var scaf=String(card.dataset.scaffold||'');\n"
            "    card.classList.toggle('is-deactivated',_isScaffoldDeactivated(scaf));\n"
            "  });\n"
            "}\n"
            "function _isPosePopupMode(){\n"
            "  try{return (new URLSearchParams(window.location.search)).get(_POSE_POPUP_QUERY_KEY)==='1';}catch(_e){return false;}\n"
            "}\n"
            "function _buildPosePopupUrl(idx,label){\n"
            "  var href=window.location.href.split('#')[0];\n"
            "  var url;\n"
            "  try{url=new URL(href);}catch(_e){return href;}\n"
            "  url.searchParams.set(_POSE_POPUP_QUERY_KEY,'1');\n"
            "  var params=new URLSearchParams();\n"
            "  params.set('action','show');\n"
            "  params.set('poseIdx',String(idx));\n"
            "  params.set('label',String(label||('Mol '+String(idx))));\n"
            "  params.set('nonce',String(Date.now()));\n"
            "  url.hash=params.toString();\n"
            "  return url.toString();\n"
            "}\n"
            "function _getPosePopupWindow(){\n"
            "  try{if(_POSE_POPUP_REF&&!_POSE_POPUP_REF.closed){return _POSE_POPUP_REF;}}catch(_e){}\n"
            "  return null;\n"
            "}\n"
            "function _sendPosePopupCommand(action,idx,label,checked){\n"
            "  var popup=_getPosePopupWindow();\n"
            "  if(!popup){return false;}\n"
            "  try{\n"
            "    popup.postMessage({channel:'rgroup-pose-popup',action:String(action||'show'),poseIdx:idx,label:String(label||('Mol '+String(idx))),checked:(typeof checked==='undefined'?null:!!checked)},'*');\n"
            "    return true;\n"
            "  }catch(_e){}\n"
            "  try{\n"
            "    var params=new URLSearchParams();\n"
            "    params.set('action',String(action||'show'));\n"
            "    params.set('poseIdx',String(idx));\n"
            "    params.set('label',String(label||('Mol '+String(idx))));\n"
            "    if(typeof checked!=='undefined'){params.set('checked',checked?'1':'0');}\n"
            "    params.set('nonce',String(Date.now()));\n"
            "    popup.location.hash=params.toString();\n"
            "    return true;\n"
            "  }catch(_e){}\n"
            "  return false;\n"
            "}\n"
            "function _runPosePopupCommand(action,idx,label,checked){\n"
            "  if(!(idx>=0)){return;}\n"
            "  document.title='Docking Visualizer';\n"
            "  if(action==='overlay'&&typeof _applyPoseOverlaySelection==='function'){_applyPoseOverlaySelection(idx,label,!!checked);return;}\n"
            "  if(typeof _showPoseFromIndex==='function'){_showPoseFromIndex(idx,label);}else if(typeof _renderPoseFromIndex==='function'){_renderPoseFromIndex(idx,label);}\n"
            "}\n"
            "function _consumePosePopupMessage(event){\n"
            "  if(!_isPosePopupMode()){return;}\n"
            "  var data=event&&event.data?event.data:null;\n"
            "  if(!data||data.channel!=='rgroup-pose-popup'){return;}\n"
            "  if(typeof _mkPosePanel==='function'){_mkPosePanel();}\n"
            "  document.body.classList.add('pose-popup-mode');\n"
            "  var panel=document.getElementById('pose-panel');\n"
            "  if(panel){panel.style.display='flex';}\n"
            "  var idx=parseInt(data.poseIdx,10);\n"
            "  var label=String(data.label||('Mol '+String(idx)));\n"
            "  _runPosePopupCommand(String(data.action||'show'),idx,label,!!data.checked);\n"
            "}\n"
            "function _syncPoseOverlayToPopup(idx,label,checked){\n"
            "  return _sendPosePopupCommand('overlay',idx,label,checked);\n"
            "}\n"
            "function _openPoseVisualizerWindow(idx,label){\n"
            "  if(_isPosePopupMode()){if(typeof _showPoseFromIndex==='function'){_showPoseFromIndex(idx,label);}else if(typeof _renderPoseFromIndex==='function'){_renderPoseFromIndex(idx,label);}return;}\n"
            "  var popupUrl=_buildPosePopupUrl(idx,label);\n"
            "  var popup=_getPosePopupWindow();\n"
            "  if(popup&&_sendPosePopupCommand('show',idx,label)){try{popup.focus();}catch(_e){}return;}\n"
            "  if(!popup){popup=window.open(popupUrl,'rgroup_pose_visualizer','popup=yes,width=1400,height=900,resizable=yes,scrollbars=yes');}\n"
            "  if(popup){\n"
            "    _POSE_POPUP_REF=popup;\n"
            "    try{popup.location.replace(popupUrl);}catch(_e){}\n"
            "    try{popup.focus();}catch(_e){}\n"
            "  }else if(typeof _showPoseFromIndex==='function'){\n"
            "    _showPoseFromIndex(idx,label);\n"
            "  }else if(typeof _renderPoseFromIndex==='function'){\n"
            "    _renderPoseFromIndex(idx,label);\n"
            "  }\n"
            "}\n"
            "function _consumePosePopupRequestFromHash(){\n"
            "  if(!_isPosePopupMode()){return;}\n"
            "  if(typeof _mkPosePanel==='function'){_mkPosePanel();}\n"
            "  document.body.classList.add('pose-popup-mode');\n"
            "  var panel=document.getElementById('pose-panel');\n"
            "  if(panel){panel.style.display='flex';}\n"
            "  var raw=String(window.location.hash||'').replace(/^#/,'');\n"
            "  if(!raw){return;}\n"
            "  var hashParams=new URLSearchParams(raw);\n"
            "  var action=hashParams.get('action')||'show';\n"
            "  var idx=parseInt(hashParams.get('poseIdx')||'-1',10);\n"
            "  var label=hashParams.get('label')||('Mol '+String(idx));\n"
            "  _runPosePopupCommand(action,idx,label,hashParams.get('checked')==='1');\n"
            "}\n"
            "function _decodeBase64ToUint8Array(base64Text){\n"
            "  var binary=window.atob(base64Text);\n"
            "  var bytes=new Uint8Array(binary.length);\n"
            "  for(var idx=0;idx<binary.length;idx+=1){bytes[idx]=binary.charCodeAt(idx);}\n"
            "  return bytes;\n"
            "}\n"
            "function _setStatusMessage(status,message){\n"
            "  if(!status){return;}\n"
            "  var textEl=status.querySelector('.search-status-line');\n"
            "  if(textEl){textEl.textContent=message;}\n"
            "  else{status.textContent=message;}\n"
            "}\n"
            "function _setSearchProgress(kind,done,total){\n"
            "  var fill=document.getElementById(kind==='exclude'?'exclude-motif-progress-fill':'structure-search-progress-fill');\n"
            "  if(!fill){return;}\n"
            "  var track=fill.parentNode;\n"
            "  if(!track){return;}\n"
            "  if(!total){track.style.display='none';fill.style.width='0%';fill.title='';return;}\n"
            "  var pct=Math.max(0,Math.min(100,Math.round((done*100)/total)));\n"
            "  track.style.display='block';\n"
            "  fill.style.width=pct+'%';\n"
            "  fill.title=String(done)+' / '+String(total);\n"
            "}\n"
            "function _setStructureSearchStatus(message,tone){\n"
            "  var status=document.getElementById('structure-search-status');\n"
            "  if(!status){return;}\n"
            "  _setStatusMessage(status,message);\n"
            "  status.dataset.tone=tone||'ready';\n"
            "  if((tone||'ready')!=='working'){_setSearchProgress('structure',0,0);}\n"
            "}\n"
            "function _setExcludeMotifStatus(message,tone){\n"
            "  var status=document.getElementById('exclude-motif-status');\n"
            "  if(!status){return;}\n"
            "  _setStatusMessage(status,message);\n"
            "  status.dataset.tone=tone||'ready';\n"
            "  if((tone||'ready')!=='working'){_setSearchProgress('exclude',0,0);}\n"
            "}\n"
            "function _svgToDataUrl(svg){\n"
            "  return 'data:image/svg+xml;charset=utf-8,'+encodeURIComponent(String(svg||''));\n"
            "}\n"
            "function _ensureRDKit(){\n"
            "  if(!_rdkitReadyPromise){\n"
            "    if(typeof window.initRDKitModule!=='function'){\n"
            "      _rdkitReadyPromise=Promise.reject(new Error('RDKit loader script is unavailable.'));\n"
            "    } else {\n"
            "      var moduleOptions={locateFile:function(){return _RDKIT_WASM_PATH;}};\n"
            "      if(_RDKIT_WASM_B64){moduleOptions.wasmBinary=_decodeBase64ToUint8Array(_RDKIT_WASM_B64);}\n"
            "      _rdkitReadyPromise=window.initRDKitModule(moduleOptions).then(function(RDKit){window.RDKit=RDKit;return RDKit;});\n"
            "    }\n"
            "  }\n"
            "  return _rdkitReadyPromise;\n"
            "}\n"
            "function _getStructureEntriesForScope(scope){\n"
            "  var key=(scope==='all_report_eligible')?'all_report_eligible':'html_subset';\n"
            "  return key==='all_report_eligible'?_STRUCTURE_SEARCH_ENTRIES_ALL:_STRUCTURE_SEARCH_ENTRIES_HTML;\n"
            "}\n"
            "function _getSearchScope(){\n"
            "  var scope=String(document.body.dataset.searchScope||_DEFAULT_SEARCH_SCOPE||'html_subset');\n"
            "  return scope==='all_report_eligible'?'all_report_eligible':'html_subset';\n"
            "}\n"
            "function _setSearchScope(scope,options){\n"
            "  var key=(scope==='all_report_eligible')?'all_report_eligible':'html_subset';\n"
            "  document.body.dataset.searchScope=key;\n"
            "  document.querySelectorAll('.search-scope-tab').forEach(function(btn){\n"
            "    var bScope=String(btn.dataset.searchScope||'html_subset');\n"
            "    var active=(bScope===key);\n"
            "    btn.classList.toggle('active',active);\n"
            "    btn.setAttribute('aria-selected',active?'true':'false');\n"
            "  });\n"
            "  if(options&&options.announce){\n"
            "    var label=(key==='all_report_eligible')?'All report-eligible molecules':'HTML page molecules subset';\n"
            "    _setStructureSearchStatus('Search scope set to '+label+'.','ready');\n"
            "  }\n"
            "}\n"
            "function _scopeLabel(scope){\n"
            "  return scope==='all_report_eligible'?'all report-eligible molecules':'HTML page subset';\n"
            "}\n"
            "function _scopeBadgeLabel(scope){\n"
            "  return scope==='all_report_eligible'?'Scope: All report-eligible':'';\n"
            "}\n"
            "function _updateStructureScopeBadge(scope){\n"
            "  var badge=document.getElementById('structure-search-scope-badge');\n"
            "  if(!badge){return;}\n"
            "  var text=_scopeBadgeLabel(scope);\n"
            "  badge.textContent=text;\n"
            "  badge.style.display=text?'inline-flex':'none';\n"
            "}\n"
            "function _ensureStructureLibrary(scope){\n"
            "  var key=(scope==='all_report_eligible')?'all_report_eligible':'html_subset';\n"
            "  if(!_structureLibraryByScope[key]){\n"
            "    _structureLibraryByScope[key]=_ensureRDKit().then(function(RDKit){\n"
            "      var library=new RDKit.SubstructLibrary();\n"
            "      _getStructureEntriesForScope(key).forEach(function(entry){library.add_trusted_smiles(entry.smiles);});\n"
            "      return {RDKit:RDKit,library:library,scope:key};\n"
            "    });\n"
            "  }\n"
            "  return _structureLibraryByScope[key];\n"
            "}\n"
            "function _warmupStructureSearchAssets(){\n"
            "  if(_structureWarmupStarted){return;}\n"
            "  _structureWarmupStarted=true;\n"
            "  _setStructureSearchStatus('Loading RDKit search assets in background…','working');\n"
            "  _ensureStructureLibrary(_getSearchScope()).then(function(){\n"
            "    _setStructureSearchStatus('Structure search ready. RDKit '+(window.RDKit?window.RDKit.version():'')+' loaded locally.','ready');\n"
            "  }).catch(function(err){\n"
            "    _setStructureSearchStatus((err&&err.message)?err.message:'Failed to load RDKit assets.','error');\n"
            "  });\n"
            "}\n"
            "function _getStructureSearchMode(){\n"
            "  var checked=document.querySelector('input[name=\"structure-search-mode\"]:checked');\n"
            "  return checked?String(checked.value||'substructure'):'substructure';\n"
            "}\n"
            "function _getStructureSearchInputText(){\n"
            "  var input=document.getElementById('structure-search-input');\n"
            "  var text=String((input&&input.value)||'').trim();\n"
            "  if(!text){return '';}\n"
            "  var firstLine=text.split(/\\r?\\n/)[0].trim();\n"
            "  var cxIdx=firstLine.indexOf(' |');\n"
            "  if(cxIdx>0){firstLine=firstLine.slice(0,cxIdx).trim();}\n"
            "  return firstLine;\n"
            "}\n"
            "function _getExcludeMotifInputText(){\n"
            "  var input=document.getElementById('exclude-motif-input');\n"
            "  var text=String((input&&input.value)||'').trim();\n"
            "  if(!text){return '';}\n"
            "  return text.split(/\\r?\\n/)[0].trim();\n"
            "}\n"
            "function _getExcludedMolIdSet(){\n"
            "  if(!_EXCLUDE_MOTIF_STATE.items.length){return null;}\n"
            "  var out=new Set();\n"
            "  _EXCLUDE_MOTIF_STATE.items.forEach(function(item){\n"
            "    if(!item||!item.matchedMolIds){return;}\n"
            "    item.matchedMolIds.forEach(function(molId){out.add(String(molId));});\n"
            "  });\n"
            "  return out;\n"
            "}\n"
            "function _memberPassesActiveFilters(member,options){\n"
            "  var opts=options||{};\n"
            "  if(!member){return false;}\n"
            "  var molId=String((member&&member.mol_id)||'');\n"
            "  if(opts.applyExclusion!==false){\n"
            "    var excluded=_getExcludedMolIdSet();\n"
            "    if(excluded&&molId&&excluded.has(molId)){return false;}\n"
            "  }\n"
            "  if(opts.applyPropFilters){\n"
            "    var activeFilters=(opts.propNames&&opts.propNames.length)?opts.propNames:(typeof _getActivePropFilterNames==='function'?_getActivePropFilterNames():[]);\n"
            "    if(activeFilters.length){\n"
            "      if(!molId){return false;}\n"
            "      if(typeof _molPassesNamedPropFilters!=='function'||!_molPassesNamedPropFilters(molId,activeFilters)){return false;}\n"
            "    }\n"
            "  }\n"
            "  if(opts.applyStructure&&_STRUCTURE_MATCH_STATE.active){\n"
            "    if(!molId||!_STRUCTURE_MATCH_STATE.matchedMolIds||!_STRUCTURE_MATCH_STATE.matchedMolIds.has(molId)){return false;}\n"
            "  }\n"
            "  return true;\n"
            "}\n"
            "function _getFilteredScaffoldMembers(scaffoldName,options){\n"
            "  var scaf=String(scaffoldName||'');\n"
            "  var exportEntry=_EXPORT[scaf]||null;\n"
            "  if(!exportEntry){return [];}\n"
            "  var opts=options||{};\n"
            "  var members=null;\n"
            "  if(opts.preferDisplayMembers&&Array.isArray(exportEntry.display_members)&&exportEntry.display_members.length){members=exportEntry.display_members;}\n"
            "  else if(Array.isArray(exportEntry.all_members)){members=exportEntry.all_members;}\n"
            "  else if(Array.isArray(exportEntry.display_members)){members=exportEntry.display_members;}\n"
            "  else{return [];}\n"
            "  return members.filter(function(member){return _memberPassesActiveFilters(member,opts);});\n"
            "}\n"
            "function _renderExcludeMotifRows(){\n"
            "  var host=document.getElementById('exclude-motif-active-list');\n"
            "  if(!host){return;}\n"
            "  if(!_EXCLUDE_MOTIF_STATE.items.length){host.innerHTML='';return;}\n"
            "  host.innerHTML=_EXCLUDE_MOTIF_STATE.items.map(function(item){\n"
            "    var imgHtml=item.previewUrl?('<img src=\"'+item.previewUrl+'\" alt=\"motif\" style=\"width:96px;height:72px;object-fit:contain;background:#fff;border:1px solid #d7e1ec;border-radius:8px;padding:4px;\"/>'):('<div style=\"width:96px;height:72px;display:flex;align-items:center;justify-content:center;background:#fff;border:1px solid #d7e1ec;border-radius:8px;color:#607489;font-size:11px;font-weight:600;\">SMARTS</div>');\n"
                "    var safeQuery=String(item.displayQuery||item.queryText||'');\n"
                "    safeQuery=safeQuery.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\"/g,'&quot;').replace(/'/g,'&#39;');\n"
            "    var scopeLabel=_scopeLabel(String(item.scope||'html_subset'));\n"
            "    return '<div style=\"display:flex;align-items:center;gap:12px;padding:10px 12px;border:1px solid #bfe3cb;border-radius:12px;background:#eefaf1;box-shadow:inset 0 0 0 1px rgba(11,110,79,0.04);\">'+imgHtml+'<div style=\"flex:1 1 auto;min-width:0;\"><div style=\"font-size:12px;font-weight:700;color:#1d6134;\">Exclusion active</div><div style=\"font-size:12px;color:#1f3551;word-break:break-word;\">'+safeQuery+'</div><div style=\"font-size:11px;color:#4f6478;margin-top:4px;\">Removes '+String(item.removedCount||0)+' molecules in '+scopeLabel+'.</div></div><button type=\"button\" onclick=\"_removeExcludeMotif('+String(item.id)+')\" style=\"border:1px solid #c4d0df;border-radius:7px;padding:5px 10px;background:#fff;cursor:pointer;font-size:12px;color:#1f3551;\">Remove</button></div>';\n"
            "  }).join('');\n"
            "}\n"
            "function _syncExcludeMotifEffects(){\n"
            "  _renderExcludeMotifRows();\n"
            "  _updateCentralMemberCountLabels();\n"
            "  _updateIncludeAllMemberLabels();\n"
            "  _updateVisibleCentralDropBadges();\n"
            "  _applyExcludeMotifToDeepDives();\n"
            "  _applyDuplicateHideToDeepDives();\n"
            "  _updateDeepDiveMemberCounts();\n"
            "  _updateVisibleCentralDropBadges();\n"
            "}\n"
            "function _removeExcludeMotif(id){\n"
            "  var target=parseInt(id,10);\n"
            "  if(!isFinite(target)){return;}\n"
            "  _EXCLUDE_MOTIF_STATE.items=_EXCLUDE_MOTIF_STATE.items.filter(function(item){return parseInt(item.id,10)!==target;});\n"
            "  _syncExcludeMotifEffects();\n"
            "  if(_EXCLUDE_MOTIF_STATE.items.length){_setExcludeMotifStatus('Active motif exclusions: '+_EXCLUDE_MOTIF_STATE.items.length+'.','success');}\n"
            "  else{_setExcludeMotifStatus('No motif exclusion is active.','ready');}\n"
            "}\n"
            "function _countScaffoldMembers(scaffoldName,options){\n"
            "  var scaf=String(scaffoldName||'');\n"
            "  var opts=options||{};\n"
            "  var members=_getFilteredScaffoldMembers(scaf,{\n"
            "    applyExclusion:opts.applyExclusion!==false,\n"
            "    applyPropFilters:opts.applyPropFilters!==false,\n"
            "    applyStructure:opts.applyStructure!==false,\n"
            "    preferDisplayMembers:!!opts.preferDisplayMembers,\n"
            "    propNames:opts.propNames||null\n"
            "  });\n"
            "  var raw=members.length;\n"
            "  var uniq=0;\n"
            "  var seen={};\n"
            "  members.forEach(function(member){\n"
            "    var key=String((member&&member.canonical_smiles)||(member&&member.smiles)||(member&&member.mol_id)||'');\n"
            "    if(!seen[key]){seen[key]=true;uniq+=1;}\n"
            "  });\n"
            "  return {raw:raw,unique:uniq};\n"
            "}\n"
            "var _centralCardMap=null;\n"
            "function _getCentralCardMap(){\n"
            "  if(!_centralCardMap){\n"
            "    _centralCardMap={};\n"
            "    (_REPORT_RENDER_PAYLOAD&&Array.isArray(_REPORT_RENDER_PAYLOAD.centralCards)?_REPORT_RENDER_PAYLOAD.centralCards:[]).forEach(function(entry){_centralCardMap[String(entry.scaffold||'')]=entry;});\n"
            "  }\n"
            "  return _centralCardMap;\n"
            "}\n"
            "function _getCentralCards(){\n"
            "  return (_REPORT_RENDER_PAYLOAD&&Array.isArray(_REPORT_RENDER_PAYLOAD.centralCards))?_REPORT_RENDER_PAYLOAD.centralCards:[];\n"
            "}\n"
            "function _getCentralCardEntry(scaffold){\n"
            "  return _getCentralCardMap()[String(scaffold||'')]||null;\n"
            "}\n"
            "function _cssEsc(value){\n"
            "  if(window.CSS&&typeof window.CSS.escape==='function'){return window.CSS.escape(String(value||''));}\n"
            "  return String(value||'').replace(/([\\\"'\\[\\]#.:,>+~*()=])/g,'\\\\$1');\n"
            "}\n"
            "function _syncVisibleScaffoldSelections(){\n"
            "  document.querySelectorAll('.scaf-checkbox').forEach(function(cb){\n"
            "    var scaf=String(cb.dataset.scaffold||'');\n"
            "    var checked=!!(typeof _isScaffoldSelected==='function'&&_isScaffoldSelected(scaf));\n"
            "    cb.checked=checked;\n"
            "  });\n"
            "  document.querySelectorAll('.idea-card,.card[id^=\"dd-\"]').forEach(function(el){\n"
            "    var scaf=String(el.dataset.scaffold||'');\n"
            "    el.classList.toggle('sel-active',!!(typeof _isScaffoldSelected==='function'&&_isScaffoldSelected(scaf)));\n"
            "  });\n"
            "  _syncDeactivatedScaffolds();\n"
            "}\n"
            "function _isDuplicateHideEnabled(){\n"
            "  var cb=document.getElementById('filter-hide-duplicates');\n"
            "  return !!(cb&&cb.checked);\n"
            "}\n"
            "function _updateCentralMemberCountLabels(){\n"
            "  var dedupe=_isDuplicateHideEnabled();\n"
            "  document.querySelectorAll('#central-idea-grid .central-members-count').forEach(function(el){\n"
            "    var card=el.closest('.idea-card');\n"
            "    var counts=_countScaffoldMembers(card?card.dataset.scaffold:'');\n"
            "    var raw=counts.raw;\n"
            "    var uniq=counts.unique;\n"
            "    el.textContent=String(dedupe?uniq:raw);\n"
            "  });\n"
            "}\n"
            "function _updateIncludeAllMemberLabels(){\n"
            "  var dedupe=_isDuplicateHideEnabled();\n"
            "  document.querySelectorAll('#central-idea-grid .all-members-label').forEach(function(el){\n"
            "    var card=el.closest('.idea-card');\n"
            "    var counts=_countScaffoldMembers(card?card.dataset.scaffold:'');\n"
            "    var raw=counts.raw;\n"
            "    var uniq=counts.unique;\n"
            "    var shown=dedupe?uniq:raw;\n"
            "    el.textContent='Include all '+String(shown)+' members in SDF';\n"
            "  });\n"
            "}\n"
            "function _applyExcludeMotifToDeepDives(){\n"
            "  var excluded=_getExcludedMolIdSet();\n"
            "  document.querySelectorAll('#deep-dive-shell .moltile').forEach(function(tile){\n"
            "    var molId=String(tile.dataset.molId||'');\n"
            "    tile.dataset.excludeHidden=(excluded&&molId&&excluded.has(molId))?'1':'0';\n"
            "  });\n"
            "}\n"
            "function _applyDuplicateHideToDeepDives(){\n"
            "  var dedupe=_isDuplicateHideEnabled();\n"
            "  document.querySelectorAll('#deep-dive-shell .card[data-scaffold]').forEach(function(card){\n"
            "    var seen={};\n"
            "    card.querySelectorAll('.moltile').forEach(function(tile,idx){\n"
            "      var propHidden=String(tile.dataset.propHidden||'0')==='1';\n"
            "      var excludeHidden=String(tile.dataset.excludeHidden||'0')==='1';\n"
            "      if(propHidden||excludeHidden){tile.style.display='none';tile.dataset.dupHidden='0';return;}\n"
            "      if(!dedupe){tile.style.display='';tile.dataset.dupHidden='0';return;}\n"
            "      var canonical=String(tile.dataset.searchCanonical||'').trim();\n"
            "      var fallback=String(tile.dataset.molId||tile.dataset.molIndex||idx);\n"
            "      var key=canonical||('fallback:'+fallback);\n"
            "      if(seen[key]){tile.style.display='none';tile.dataset.dupHidden='1';return;}\n"
            "      seen[key]=true;tile.style.display='';tile.dataset.dupHidden='0';\n"
            "    });\n"
            "  });\n"
            "}\n"
            "function _updateDeepDiveMemberCounts(){\n"
            "  document.querySelectorAll('#deep-dive-shell .card[data-scaffold]').forEach(function(card){\n"
            "    var countEl=card.querySelector('.dd-members-count');\n"
            "    if(!countEl){return;}\n"
            "    var counts=_countScaffoldMembers(card.dataset.scaffold||'');\n"
            "    countEl.textContent=String(_isDuplicateHideEnabled()?counts.unique:counts.raw);\n"
            "  });\n"
            "}\n"
            "function _getCentralPageSize(totalCount){\n"
            "  var sizeEl=document.getElementById('central-page-size');\n"
            "  var raw=String((sizeEl&&sizeEl.value)||_CENTRAL_RENDER_STATE.pageSize||'25').toLowerCase();\n"
            "  if(raw==='all'){return Math.max(1,totalCount||1);}\n"
            "  var parsed=parseInt(raw,10);\n"
            "  if(!isFinite(parsed)||parsed<=0){parsed=25;}\n"
            "  return parsed;\n"
            "}\n"
            "function _downloadScaffoldList(kind){\n"
            "  var rows=[];\n"
            "  if(kind==='red'){rows=_CENTRAL_RENDER_STATE.lastRedNames||[];}\n"
            "  else if(kind==='green'){rows=_CENTRAL_RENDER_STATE.lastGreenNames||[];}\n"
            "  if(!rows.length){return;}\n"
            "  var text=rows.join('\\n')+'\\n';\n"
            "  var blob=new Blob([text],{type:'text/plain'});\n"
            "  var a=document.createElement('a');\n"
            "  a.href=URL.createObjectURL(blob);\n"
            "  a.download=(kind==='red'?'top15_red_scaffolds.txt':'green_scaffolds.txt');\n"
            "  a.click();\n"
            "}\n"
            "function _updateHighlightListing(){\n"
            "  var host=document.getElementById('highlight-listing');\n"
            "  if(!host){return;}\n"
            "  var redOn=!!(document.getElementById('filter-red-highlight')||{}).checked;\n"
            "  var greenOn=!!(document.getElementById('filter-green-highlight')||{}).checked;\n"
            "  var allCards=_getCentralCards();\n"
            "  var redCards=allCards.filter(function(entry){return !!entry.isTop15;}).sort(function(a,b){return (b.nMembers||0)-(a.nMembers||0);});\n"
            "  var greenCards=allCards.filter(function(entry){return !!entry.isHighDistance;});\n"
            "  _CENTRAL_RENDER_STATE.lastRedNames=redCards.map(function(entry){return String(entry.scaffold||'');});\n"
            "  _CENTRAL_RENDER_STATE.lastGreenNames=greenCards.map(function(entry){return String(entry.scaffold||'');});\n"
            "  var parts=['<div style=\\\"margin-bottom:4px;\\\"><b>All scaffolds (full dataset):</b> Red (Top-15) = '+redCards.length+'&nbsp;&nbsp;|&nbsp;&nbsp;Green (high-distance) = '+greenCards.length+'</div>'];\n"
            "  if(redOn){\n"
            "    var rn=_CENTRAL_RENDER_STATE.lastRedNames;\n"
            "    parts.push('<div style=\\\"margin-top:6px;\\\"><span class=\\\"tag red\\\">Top-15 most populous clusters</span> ('+rn.length+' scaffolds):<div style=\\\"font-size:11px;max-height:120px;overflow-y:auto;border:1px solid #d6dee8;padding:4px 6px;margin:4px 0;border-radius:4px;background:#fff;\\\">'+rn.join('<br/>')+'</div><a href=\\\"javascript:void(0)\\\" onclick=\\\"_downloadScaffoldList(\\\'red\\\')\\\">Download list</a></div>');\n"
            "  }\n"
            "  if(greenOn){\n"
            "    var gn=_CENTRAL_RENDER_STATE.lastGreenNames;\n"
            "    parts.push('<div style=\\\"margin-top:6px;\\\"><span class=\\\"tag green\\\">High-distance scaffolds</span> ('+gn.length+' scaffolds):<div style=\\\"font-size:11px;max-height:200px;overflow-y:auto;border:1px solid #d6dee8;padding:4px 6px;margin:4px 0;border-radius:4px;background:#fff;\\\">'+gn.join('<br/>')+'</div><a href=\\\"javascript:void(0)\\\" onclick=\\\"_downloadScaffoldList(\\\'green\\\')\\\">Download list</a></div>');\n"
            "  }\n"
            "  if(!redOn&&!greenOn){parts.push('<div>Enable Red and/or Green highlight checkboxes to list scaffolds by category.</div>');}\n"
            "  host.innerHTML=parts.join('');\n"
            "}\n"
            "function _updateCentralPager(totalPages,totalCount){\n"
            "  _CENTRAL_RENDER_STATE.lastTotalPages=totalPages;\n"
            "  var controls=[\n"
            "    {id:'central-page-first',kind:'first'},{id:'central-page-prev',kind:'prev'},{id:'central-page-next',kind:'next'},{id:'central-page-last',kind:'last'},\n"
            "    {id:'central-page-first-bottom',kind:'first'},{id:'central-page-prev-bottom',kind:'prev'},{id:'central-page-next-bottom',kind:'next'},{id:'central-page-last-bottom',kind:'last'}\n"
            "  ];\n"
            "  controls.forEach(function(item){\n"
            "    var btn=document.getElementById(item.id);\n"
            "    if(!btn){return;}\n"
            "    if(item.kind==='first'||item.kind==='prev'){btn.disabled=_CENTRAL_RENDER_STATE.page<=0||totalCount===0;}\n"
            "    else{btn.disabled=_CENTRAL_RENDER_STATE.page>=totalPages-1||totalCount===0;}\n"
            "  });\n"
            "  ['central-page-status','central-page-status-bottom'].forEach(function(id){\n"
            "    var status=document.getElementById(id);\n"
            "    if(!status){return;}\n"
            "    if(totalCount===0){status.textContent='No matching cards';}\n"
            "    else{status.textContent='Page '+(_CENTRAL_RENDER_STATE.page+1)+' of '+totalPages+' ('+totalCount+' scaffold'+(totalCount===1?'':'s')+')';}\n"
            "  });\n"
            "}\n"
            "function _formatCountLabel(count,singular,plural){\n"
            "  return String(count)+' '+(count===1?singular:plural);\n"
            "}\n"
            "function _getActivePropFilterNames(){\n"
            "  return typeof _PROP_FILTER_STATE==='undefined'?[]:Object.keys(_PROP_FILTER_STATE);\n"
            "}\n"
            "function _getMolPropsData(){\n"
            "  if(_MOL_PROPS_DATA===null){\n"
            "    _MOL_PROPS_DATA=_MOL_PROPS_DATA_JSON?JSON.parse(_MOL_PROPS_DATA_JSON):{};\n"
            "  }\n"
            "  return _MOL_PROPS_DATA||{};\n"
            "}\n"
            "function _getScaffoldMolMap(){\n"
            "  if(_SCAFFOLD_MOL_MAP===null){\n"
            "    _SCAFFOLD_MOL_MAP=_SCAFFOLD_MOL_MAP_JSON?JSON.parse(_SCAFFOLD_MOL_MAP_JSON):{};\n"
            "  }\n"
            "  return _SCAFFOLD_MOL_MAP||{};\n"
            "}\n"
            "function _molPassesNamedPropFilters(molId,propNames){\n"
            "  var filters=propNames||_getActivePropFilterNames();\n"
            "  if(!filters.length){return true;}\n"
            "  var propsById=_getMolPropsData();\n"
            "  var props=propsById[molId];\n"
            "  if(!props){return false;}\n"
            "  return filters.every(function(p){\n"
            "    var pIdx=_PROP_NAMES.indexOf(p);\n"
            "    if(pIdx<0){return true;}\n"
            "    var range=_PROP_FILTER_STATE[p];\n"
            "    if(!range){return true;}\n"
            "    var val=props[pIdx];\n"
            "    return val!==null&&val!==undefined&&val>=range.min&&val<=range.max;\n"
            "  });\n"
            "}\n"
            "function _getScaffoldPropFilterStats(scaffoldName,propNames){\n"
            "  var scaffoldMap=_getScaffoldMolMap();\n"
            "  var members=(scaffoldMap[scaffoldName])||[];\n"
            "  var totalCount=members.length;\n"
            "  var filters=propNames||_getActivePropFilterNames();\n"
            "  if(!filters.length||!totalCount){return {totalCount:totalCount,keptCount:totalCount,removedCount:0,removedPct:0};}\n"
            "  var keptCount=0;\n"
            "  members.forEach(function(molId){if(_molPassesNamedPropFilters(molId,filters)){keptCount+=1;}});\n"
            "  var removedCount=Math.max(0,totalCount-keptCount);\n"
            "  var removedPct=totalCount?Math.round((removedCount*100)/totalCount):0;\n"
            "  return {totalCount:totalCount,keptCount:keptCount,removedCount:removedCount,removedPct:removedPct};\n"
            "}\n"
            "function _computePropImpactSummary(baseCards,activeFilters){\n"
            "  var cards=baseCards||[];\n"
            "  var summary={activeFilters:activeFilters||[],scaffoldsBase:0,scaffoldsRemoved:0,moleculesBase:0,moleculesRemoved:0,scaffoldStats:{},perFilter:[]};\n"
            "  summary.scaffoldsBase=cards.length;\n"
            "  cards.forEach(function(entry){\n"
            "    var scaf=String(entry.scaffold||'');\n"
            "    var stats=_getScaffoldPropFilterStats(scaf,summary.activeFilters);\n"
            "    summary.scaffoldStats[scaf]=stats;\n"
            "    summary.moleculesBase+=stats.totalCount;\n"
            "    summary.moleculesRemoved+=stats.removedCount;\n"
            "    if(summary.activeFilters.length&&stats.keptCount===0){summary.scaffoldsRemoved+=1;}\n"
            "  });\n"
            "  summary.activeFilters.forEach(function(propName){\n"
            "    var item={propName:propName,label:(_PROP_LABELS&&_PROP_LABELS[propName])||propName,scaffoldsRemoved:0,moleculesRemoved:0};\n"
            "    cards.forEach(function(entry){\n"
            "      var stats=_getScaffoldPropFilterStats(String(entry.scaffold||''),[propName]);\n"
            "      item.moleculesRemoved+=stats.removedCount;\n"
            "      if(stats.keptCount===0){item.scaffoldsRemoved+=1;}\n"
            "    });\n"
            "    summary.perFilter.push(item);\n"
            "  });\n"
            "  return summary;\n"
            "}\n"
            "function _updateCentralFilterImpactSummary(){\n"
            "  var wrap=document.getElementById('central-filter-impact-wrap');\n"
            "  var summaryEl=document.getElementById('central-filter-impact-summary');\n"
            "  var breakdownEl=document.getElementById('central-filter-impact-breakdown');\n"
            "  if(!wrap||!summaryEl||!breakdownEl){return;}\n"
            "  var summary=_CENTRAL_RENDER_STATE.propImpactSummary;\n"
            "  if(!summary||!summary.activeFilters||!summary.activeFilters.length){\n"
            "    wrap.style.display='none';\n"
            "    summaryEl.textContent='';\n"
            "    breakdownEl.textContent='';\n"
            "    return;\n"
            "  }\n"
            "  wrap.style.display='';\n"
            "  summaryEl.textContent='Filters removed '+_formatCountLabel(summary.scaffoldsRemoved,'scaffold','scaffolds')+', '+_formatCountLabel(summary.moleculesRemoved,'molecule','molecules')+'.';\n"
            "  breakdownEl.textContent=summary.perFilter.map(function(item){\n"
            "    return item.label+' filter: -'+item.scaffoldsRemoved+' scaffold'+(item.scaffoldsRemoved===1?'':'s')+', -'+item.moleculesRemoved+' mol'+(item.moleculesRemoved===1?'':'s');\n"
            "  }).join(' | ');\n"
            "}\n"
            "function _getScaffoldImpactSummary(scaf){\n"
            "  var exportEntry=_EXPORT[scaf]||null;\n"
            "  var baseCount=(exportEntry&&Array.isArray(exportEntry.all_members))?exportEntry.all_members.length:0;\n"
            "  var stats=(_CENTRAL_RENDER_STATE.scaffoldPropStats||{})[scaf]||null;\n"
            "  var parts=[];\n"
            "  var titles=[];\n"
            "  if(baseCount>0&&_STRUCTURE_MATCH_STATE&&_STRUCTURE_MATCH_STATE.active){\n"
            "    var structureCounts=_countScaffoldMembers(scaf,{applyExclusion:false,applyPropFilters:false,applyStructure:true});\n"
            "    if(structureCounts.raw>0){\n"
            "      parts.push('search '+structureCounts.raw+'/'+baseCount);\n"
            "      titles.push(structureCounts.raw+' of '+baseCount+' molecules matched the active structure search');\n"
            "    }\n"
            "  }\n"
            "  if(baseCount>0&&_EXCLUDE_MOTIF_STATE&&_EXCLUDE_MOTIF_STATE.items.length){\n"
            "    var keepAfterExclude=_countScaffoldMembers(scaf,{applyExclusion:true,applyPropFilters:false,applyStructure:false});\n"
            "    var excludeRemoved=Math.max(0,baseCount-keepAfterExclude.raw);\n"
            "    if(excludeRemoved>0){\n"
            "      parts.push('motif -'+excludeRemoved);\n"
            "      titles.push(excludeRemoved+' of '+baseCount+' molecules excluded by active motifs');\n"
            "    }\n"
            "  }\n"
            "  if(stats&&stats.totalCount&&stats.removedCount>0){\n"
            "    parts.push('prop -'+stats.removedCount);\n"
            "    titles.push(stats.removedCount+' of '+stats.totalCount+' molecules filtered by property ranges');\n"
            "  }\n"
            "  return {parts:parts,titles:titles};\n"
            "}\n"
            "function _updateVisibleCentralDropBadges(){\n"
            "  document.querySelectorAll('#central-idea-grid .central-drop-badge').forEach(function(el){\n"
            "    var scaf=String(el.dataset.scaffold||'');\n"
            "    var summary=_getScaffoldImpactSummary(scaf);\n"
            "    if(!summary.parts.length){\n"
            "      el.style.display='none';\n"
            "      el.textContent='';\n"
            "      el.title='';\n"
            "      return;\n"
            "    }\n"
            "    el.style.display='';\n"
            "    el.textContent=summary.parts.join(' | ');\n"
            "    el.title=summary.titles.join(' | ');\n"
            "  });\n"
            "  document.querySelectorAll('#central-idea-grid .scaffold-impact-summary').forEach(function(el){\n"
            "    var scaf=String(el.dataset.scaffold||'');\n"
            "    var summary=_getScaffoldImpactSummary(scaf);\n"
            "    if(!summary.parts.length){el.textContent='';el.title='';return;}\n"
            "    el.textContent='['+summary.parts.join(' | ')+']';\n"
            "    el.title=summary.titles.join(' | ');\n"
            "  });\n"
            "  document.querySelectorAll('#deep-dive-shell .scaffold-impact-summary').forEach(function(el){\n"
            "    var scaf=String(el.dataset.scaffold||'');\n"
            "    var summary=_getScaffoldImpactSummary(scaf);\n"
            "    if(!summary.parts.length){el.textContent='';el.title='';return;}\n"
            "    el.textContent='['+summary.parts.join(' | ')+']';\n"
            "    el.title=summary.titles.join(' | ');\n"
            "  });\n"
            "}\n"
            "function _getDeepDiveTpl(name){\n"
            "  return document.getElementById('dd-tpl-'+String(name||''));\n"
            "}\n"
            "function _renderDeepDivesForVisible(scaffolds){\n"
            "  var shell=document.getElementById('deep-dive-shell');\n"
            "  if(!shell){return;}\n"
            "  var priorHtml=shell.innerHTML;\n"
            "  var priorClassName=shell.className;\n"
            "  try{\n"
            "    var list=(scaffolds||[]).filter(function(name){return !_isScaffoldDeactivated(name)&&!!_getDeepDiveTpl(name);});\n"
            "    if(!list.length){\n"
            "      shell.className='lazy-placeholder';\n"
            "      shell.innerHTML='No deep-dive content is available for the currently visible central scaffolds.';\n"
            "      return;\n"
            "    }\n"
            "    shell.className='';\n"
            "    shell.innerHTML=list.map(function(name){var tpl=_getDeepDiveTpl(name);return tpl?tpl.innerHTML:'';}).join('');\n"
            "    if(typeof _bindPoseTileClicks==='function'){_bindPoseTileClicks(shell);}\n"
            "    if(typeof _bindOverlayToggles==='function'){_bindOverlayToggles();}\n"
            "    if(typeof _bindMemberToggles==='function'){_bindMemberToggles();}\n"
            "    shell.querySelectorAll('.moltile').forEach(function(tile){tile.dataset.propHidden='0';tile.dataset.dupHidden='0';});\n"
            "    _syncVisibleScaffoldSelections();\n"
            "    _applyStructureHighlighting();\n"
            "    _applyExcludeMotifToDeepDives();\n"
            "    _applyDuplicateHideToDeepDives();\n"
            "    _updateDeepDiveMemberCounts();\n"
            "    _syncDeactivatedScaffolds();\n"
            "    _updateVisibleCentralDropBadges();\n"
            "  }catch(err){\n"
            "    console.error('Deep dive render failed:',err);\n"
            "    shell.className=priorClassName;\n"
            "    shell.innerHTML=priorHtml;\n"
            "    if(typeof _setStructureSearchStatus==='function'){_setStructureSearchStatus('Deep dive render failed, but the previous content was preserved.','error');}\n"
            "  }\n"
            "}\n"
            "function _renderCentralPage(){\n"
            "  var grid=document.getElementById('central-idea-grid');\n"
            "  if(!grid){return;}\n"
            "  var priorHtml=grid.innerHTML;\n"
            "  try{\n"
            "    // Performance: slice the names array FIRST, then look up only the page's entries (O(1) each).\n"
            "    // Do NOT map all filteredScaffolds to entries — that was O(N*N) via _getCentralCardEntry's old linear find.\n"
            "    var filteredScafs=_CENTRAL_RENDER_STATE.filteredScaffolds;\n"
            "    var totalCount=filteredScafs.length;\n"
            "    var pageSize=_getCentralPageSize(totalCount||1);\n"
            "    _CENTRAL_RENDER_STATE.pageSize=pageSize;\n"
            "    var totalPages=Math.max(1,Math.ceil(totalCount/pageSize));\n"
            "    if(_CENTRAL_RENDER_STATE.page>=totalPages){_CENTRAL_RENDER_STATE.page=Math.max(0,totalPages-1);}\n"
            "    var start=_CENTRAL_RENDER_STATE.page*pageSize;\n"
            "    var sliceScafs=filteredScafs.slice(start,start+pageSize);\n"
            "    var sliceEntries=sliceScafs.map(function(scaf){return _getCentralCardEntry(scaf);}).filter(Boolean);\n"
            "    grid.innerHTML=sliceEntries.length?sliceEntries.map(function(entry){return entry.html;}).join(''):\"<div class='lazy-placeholder'>No central scaffolds match the current filters.</div>\";\n"
            "    if(typeof _bindPoseTileClicks==='function'){_bindPoseTileClicks(grid);}\n"
            "    if(typeof _bindOverlayToggles==='function'){_bindOverlayToggles();}\n"
            "    if(typeof _bindMemberToggles==='function'){_bindMemberToggles();}\n"
            "    _updateCentralMemberCountLabels();\n"
            "    _updateIncludeAllMemberLabels();\n"
            "    _updateVisibleCentralDropBadges();\n"
            "    _syncVisibleScaffoldSelections();\n"
            "    Object.keys(_STARS||{}).forEach(function(name){if(typeof _applyStarUI==='function'){_applyStarUI(name);}});\n"
            "    _syncDeactivatedScaffolds();\n"
            "    _applyStructureHighlighting();\n"
            "    _renderDeepDivesForVisible(sliceScafs);\n"
            "    if(typeof _applyPropFilterToDeepDives==='function'){_applyPropFilterToDeepDives();}\n"
            "    _applyDuplicateHideToDeepDives();\n"
            "    _updateDeepDiveMemberCounts();\n"
            "    _updateVisibleCentralDropBadges();\n"
            "    _updateCentralFilterImpactSummary();\n"
            "    _updateCentralPager(totalPages,totalCount);\n"
            "  }catch(err){\n"
            "    console.error('Central render failed:',err);\n"
            "    grid.innerHTML=priorHtml;\n"
            "    if(typeof _setStructureSearchStatus==='function'){_setStructureSearchStatus('Central render failed, but the previous content was preserved.','error');}\n"
            "  }\n"
            "}\n"
            "function scrollCentralCardIntoView(scaffold){\n"
            "  var idx=_CENTRAL_RENDER_STATE.filteredScaffolds.indexOf(String(scaffold||''));\n"
            "  if(idx<0){return;}\n"
            "  var pageSize=_getCentralPageSize(Math.max(1,_CENTRAL_RENDER_STATE.filteredScaffolds.length));\n"
            "  _CENTRAL_RENDER_STATE.page=Math.floor(idx/pageSize);\n"
            "  _renderCentralPage();\n"
            "  var target=document.querySelector('#central-idea-grid .idea-card[data-scaffold=\\\"'+_cssEsc(String(scaffold||''))+'\\\"]');\n"
            "  if(target){target.scrollIntoView({behavior:'smooth',block:'center'});}\n"
            "}\n"
            "function openDeepDive(scaffold){\n"
            "  var scaf=String(scaffold||'');\n"
            "  if(_isScaffoldDeactivated(scaf)){return;}\n"
            "  var idx=_CENTRAL_RENDER_STATE.filteredScaffolds.indexOf(scaf);\n"
            "  if(idx>=0){\n"
            "    var pageSize=_getCentralPageSize(Math.max(1,_CENTRAL_RENDER_STATE.filteredScaffolds.length));\n"
            "    _CENTRAL_RENDER_STATE.page=Math.floor(idx/pageSize);\n"
            "  }\n"
            "  _CENTRAL_RENDER_STATE.activeDeepDive=scaf;\n"
            "  _renderCentralPage();\n"
            "  var target=document.querySelector('#deep-dive-shell .card[data-scaffold=\\\"'+_cssEsc(scaf)+'\\\"]');\n"
            "  if(target){target.scrollIntoView({behavior:'smooth',block:'start'});}\n"
            "}\n"
            "function _recomputeCentralCards(resetPage){\n"
            "  var qEl=document.getElementById('central-scaffold-search');\n"
            "  var redEl=document.getElementById('filter-red-highlight');\n"
            "  var greenEl=document.getElementById('filter-green-highlight');\n"
            "  var q=String((qEl&&qEl.value)||'').trim().toLowerCase();\n"
            "  var qNorm=_normalizeScaffoldToken(q);\n"
            "  var red=!!(redEl&&redEl.checked);\n"
            "  var green=!!(greenEl&&greenEl.checked);\n"
            "  var selectedResidues=_getSelectedHBondResidues();\n"
            "  var baseCards=_getCentralCards().filter(function(entry){\n"
            "    var scaf=String(entry.scaffold||'');\n"
            "    var scafLo=scaf.toLowerCase();\n"
            "    var scafNorm=_normalizeScaffoldToken(scaf);\n"
            "    var passQuery=!q||scafLo.indexOf(q)>=0||scafNorm.indexOf(qNorm)>=0;\n"
            "    var passHBond=true;\n"
            "    if(selectedResidues.length){\n"
            "      var hits=Array.isArray(_HBOND_BY_SCAFFOLD[scaf])?_HBOND_BY_SCAFFOLD[scaf]:[];\n"
            "      passHBond=selectedResidues.every(function(res){return hits.indexOf(res)>=0;});\n"
            "    }\n"
            "    var passStructure=true;\n"
            "    if(_STRUCTURE_MATCH_STATE.active){passStructure=!!(_STRUCTURE_MATCH_STATE.matchedScaffolds&&_STRUCTURE_MATCH_STATE.matchedScaffolds.has(scaf));}\n"
            "    return passQuery&&passHBond&&passStructure;\n"
            "  });\n"
            "  var activePropFilters=_getActivePropFilterNames();\n"
            "  _CENTRAL_RENDER_STATE.propImpactSummary=_computePropImpactSummary(baseCards,activePropFilters);\n"
            "  _CENTRAL_RENDER_STATE.scaffoldPropStats=_CENTRAL_RENDER_STATE.propImpactSummary.scaffoldStats||{};\n"
            "  var cards=baseCards.filter(function(entry){\n"
            "    var stats=_CENTRAL_RENDER_STATE.scaffoldPropStats[String(entry.scaffold||'')];\n"
            "    return !activePropFilters.length||!!(stats&&stats.keptCount>0);\n"
            "  });\n"
            "  _updateHighlightListing();\n"
            "  cards.sort(function(a,b){\n"
            "    if(red||green){\n"
            "      var pa=(red&&a.isTop15?2:0)+(green&&a.isHighDistance?1:0);\n"
            "      var pb=(red&&b.isTop15?2:0)+(green&&b.isHighDistance?1:0);\n"
            "      if(pb!==pa){return pb-pa;}\n"
            "      if(pa>0&&(b.nMembers||0)!==(a.nMembers||0)){return (b.nMembers||0)-(a.nMembers||0);}\n"
            "    }\n"
            "    if(_CENTRAL_RENDER_STATE.sortMode==='starred'){\n"
            "      var sa=_STARS[a.scaffold]||0;\n"
            "      var sb=_STARS[b.scaffold]||0;\n"
            "      if(sb!==sa){return sb-sa;}\n"
            "      if((b.nMembers||0)!==(a.nMembers||0)){return (b.nMembers||0)-(a.nMembers||0);}\n"
            "    }\n"
            "    return (a.order||0)-(b.order||0);\n"
            "  });\n"
            "  _CENTRAL_RENDER_STATE.filteredScaffolds=cards.map(function(entry){return String(entry.scaffold||'');});\n"
            "  if(resetPage){_CENTRAL_RENDER_STATE.page=0;}\n"
            "  _renderCentralPage();\n"
            "}\n"
            "if(_isPosePopupMode()){window.addEventListener('hashchange',_consumePosePopupRequestFromHash);window.addEventListener('load',_consumePosePopupRequestFromHash);window.addEventListener('message',_consumePosePopupMessage);}\n"
            "function _clearStructureHighlighting(){\n"
            "  document.querySelectorAll('#central-idea-grid .idea-card,.card[id^=\"dd-\"]').forEach(function(el){\n"
            "    el.classList.remove('structure-match-active');\n"
            "  });\n"
            "}\n"
            "function _applyStructureHighlighting(){\n"
            "  _clearStructureHighlighting();\n"
            "  if(!_STRUCTURE_MATCH_STATE.active||!_STRUCTURE_MATCH_STATE.matchedScaffolds){return;}\n"
            "  document.querySelectorAll('#central-idea-grid .idea-card').forEach(function(card){\n"
            "    var scaf=String(card.dataset.scaffold||'');\n"
            "    if(_STRUCTURE_MATCH_STATE.matchedScaffolds.has(scaf)){card.classList.add('structure-match-active');}\n"
            "  });\n"
            "  document.querySelectorAll('.card[id^=\"dd-\"]').forEach(function(card){\n"
            "    var scaf=String(card.dataset.scaffold||'');\n"
            "    if(_STRUCTURE_MATCH_STATE.matchedScaffolds.has(scaf)){card.classList.add('structure-match-active');}\n"
            "  });\n"
            "}\n"
            "function _resetStructureSearch(options){\n"
            "  _STRUCTURE_MATCH_STATE.active=false;\n"
            "  _STRUCTURE_MATCH_STATE.scope=_getSearchScope();\n"
            "  _STRUCTURE_MATCH_STATE.matchedScaffolds=null;\n"
            "  _STRUCTURE_MATCH_STATE.matchedMolIds=null;\n"
            "  _STRUCTURE_MATCH_STATE.lastQuery='';\n"
            "  if(options&&options.clearInput){\n"
            "    var input=document.getElementById('structure-search-input');\n"
            "    if(input){input.value='';}\n"
            "  }\n"
            "  _applyCentralFilters();\n"
            "  _setStructureSearchStatus('Structure filter cleared.','ready');\n"
            "}\n"
            "function _resetExcludeMotif(options){\n"
            "  _EXCLUDE_MOTIF_STATE.items=[];\n"
            "  _EXCLUDE_MOTIF_STATE.lastQuery='';\n"
            "  if(options&&options.clearInput){\n"
            "    var input=document.getElementById('exclude-motif-input');\n"
            "    if(input){input.value='';}\n"
            "  }\n"
            "  _syncExcludeMotifEffects();\n"
            "  _setExcludeMotifStatus('No motif exclusion is active.','ready');\n"
            "}\n"
            "async function _runExcludeMotif(){\n"
            "  _setExcludeMotifStatus('Preparing exclusion query.','working');\n"
            "  try {\n"
            "    var scope=_getSearchScope();\n"
            "    var originalInput=_getExcludeMotifInputText();\n"
            "    var inputText=originalInput;\n"
            "    if(!inputText){_setExcludeMotifStatus('Paste a SMILES or SMARTS string first.','error');return;}\n"
            "    var RDKit=await _ensureRDKit();\n"
            "    var scopeEntries=_getStructureEntriesForScope(scope);\n"
            "    var queryMol=null;\n"
            "    var previewUrl='';\n"
            "    try {\n"
            "      var srcMol=RDKit.get_mol(inputText);\n"
            "      if(srcMol&&srcMol.is_valid()){\n"
            "        try{\n"
            "          try{previewUrl=_svgToDataUrl(srcMol.get_svg());}catch(_svgErr){}\n"
            "          var querySmarts=String(srcMol.get_smarts()||'').trim();\n"
            "          if(!querySmarts){throw new Error('Could not derive an exclusion query from the pasted SMILES.');}\n"
            "          queryMol=RDKit.get_qmol(querySmarts);\n"
            "          if(!queryMol||!queryMol.is_valid()){throw new Error('Could not build a valid substructure exclusion from the pasted SMILES.');}\n"
            "          inputText=querySmarts;\n"
            "        }finally{srcMol.delete();}\n"
            "      } else {\n"
            "        if(srcMol){srcMol.delete();}\n"
            "        queryMol=RDKit.get_qmol(inputText);\n"
            "        if(!queryMol||!queryMol.is_valid()){\n"
            "          if(queryMol){queryMol.delete();queryMol=null;}\n"
            "          throw new Error('Paste a valid SMARTS or SMILES string for motif exclusion.');\n"
            "        }\n"
            "        try{previewUrl=_svgToDataUrl(queryMol.get_svg());}catch(_qSvgErr){}\n"
            "      }\n"
            "      var matchedEntries=await _collectMatchesWithProgress(scopeEntries,scope,'substructure',queryMol,'','',RDKit,'exclude');\n"
            "      var matchedMolIds=new Set();\n"
            "      matchedEntries.forEach(function(entry){matchedMolIds.add(String(entry.mol_id||''));});\n"
            "      _EXCLUDE_MOTIF_STATE.lastQuery=inputText;\n"
            "      if(matchedMolIds.size===0){\n"
            "        _setExcludeMotifStatus('No molecules matched the exclusion motif in '+_scopeLabel(scope)+'.','working');\n"
            "      } else {\n"
            "        var duplicate=_EXCLUDE_MOTIF_STATE.items.some(function(item){return String(item.queryText||'')===String(inputText||'')&&String(item.scope||'')===String(scope||'');});\n"
            "        if(duplicate){\n"
            "          _setExcludeMotifStatus('That motif is already active for '+_scopeLabel(scope)+'.','working');\n"
            "        } else {\n"
            "          _EXCLUDE_MOTIF_STATE.items.push({id:_EXCLUDE_MOTIF_STATE.nextId++,scope:scope,queryText:inputText,displayQuery:originalInput,previewUrl:previewUrl,matchedMolIds:matchedMolIds,removedCount:matchedMolIds.size});\n"
            "          _setExcludeMotifStatus('Added exclusion motif removing '+matchedMolIds.size+' molecules in '+_scopeLabel(scope)+'. Active motifs: '+_EXCLUDE_MOTIF_STATE.items.length+'.','success');\n"
            "        }\n"
            "      }\n"
            "      _syncExcludeMotifEffects();\n"
            "    } finally { if(queryMol){queryMol.delete();} }\n"
            "  } catch (err) {\n"
            "    _setExcludeMotifStatus((err&&err.message)?err.message:'Motif exclusion failed.','error');\n"
            "  }\n"
            "}\n"
            "async function _getExactCanonicalFromText(text){\n"
            "  if(/\\[#[0-9]/.test(text)||/\\[!/.test(text)){\n"
            "    throw new Error('That looks like a SMARTS pattern. Switch to Substructure search for SMARTS queries, or paste a plain SMILES for exact molecule match.');\n"
            "  }\n"
            "  var RDKit=await _ensureRDKit();\n"
            "  var mol=RDKit.get_mol(text);\n"
            "  try {\n"
            "    if(!mol||!mol.is_valid()){throw new Error('Exact match requires a valid SMILES string.');}\n"
            "    var canonical=String(mol.get_smiles()||'').trim();\n"
            "    if(!canonical){throw new Error('Could not canonicalize the pasted SMILES string.');}\n"
            "    return canonical;\n"
            "  } finally { if(mol){mol.delete();} }\n"
            "}\n"
            "async function _collectMatchesWithProgress(scopeEntries,scope,mode,queryMol,queryCanonical,queryCanonicalNoStereo,RDKit,purpose){\n"
            "  var total=scopeEntries.length;\n"
            "  var matchedEntries=[];\n"
            "  if(!total){return matchedEntries;}\n"
            "  var chunkSize=500;\n"
            "  var label=(purpose==='exclude')?'Applying exclusion motif':'Searching';\n"
            "  var setStatus=(purpose==='exclude')?_setExcludeMotifStatus:_setStructureSearchStatus;\n"
            "  for(var start=0;start<total;start+=chunkSize){\n"
            "    var end=Math.min(total,start+chunkSize);\n"
            "    setStatus(label+' '+end+'/'+total+' molecules in '+_scopeLabel(scope)+'...','working');\n"
            "    _setSearchProgress(purpose,end,total);\n"
            "    var chunk=scopeEntries.slice(start,end);\n"
            "    if(mode==='exact'){\n"
            "      for(var i=0;i<chunk.length;i++){\n"
            "        var entry=chunk[i];\n"
            "        if(!entry){continue;}\n"
            "        var exactMatch=(entry.exact_canonical===queryCanonical)||(queryCanonicalNoStereo&&(entry.exact_canonical_nostereo===queryCanonicalNoStereo||entry.exact_canonical===queryCanonicalNoStereo));\n"
            "        if(exactMatch){matchedEntries.push(entry);}\n"
            "      }\n"
            "    } else {\n"
            "      var library=new RDKit.SubstructLibrary();\n"
            "      chunk.forEach(function(entry){if(entry&&entry.smiles){library.add_trusted_smiles(entry.smiles);}});\n"
            "      var hitIndexes=JSON.parse(library.get_matches(queryMol)||'[]');\n"
            "      hitIndexes.forEach(function(idx){var entry=chunk[idx];if(entry){matchedEntries.push(entry);}});\n"
            "    }\n"
            "    await new Promise(function(resolve){window.setTimeout(resolve,0);});\n"
            "  }\n"
            "  return matchedEntries;\n"
            "}\n"
            "async function _runStructureSearch(){\n"
            "  _setStructureSearchStatus('Preparing structure search.','working');\n"
            "  try {\n"
            "    var scope=_getSearchScope();\n"
            "    var scopeEntries=_getStructureEntriesForScope(scope);\n"
            "    var mode=_getStructureSearchMode();\n"
            "    var inputText=_getStructureSearchInputText();\n"
            "    if(!inputText){_setStructureSearchStatus('Paste a SMILES or SMARTS string first.','error');return;}\n"
            "    var matchedEntries=[];\n"
            "    if(mode==='exact'){\n"
            "      var canonical=await _getExactCanonicalFromText(inputText);\n"
            "      if(!canonical){throw new Error('Could not export a valid exact-match query.');}\n"
            "      matchedEntries=scopeEntries.filter(function(entry){return entry.exact_canonical===canonical;});\n"
            "      if(matchedEntries.length===0){\n"
            "        var RDKit=await _ensureRDKit();\n"
            "        var noStMol=null;\n"
            "        try{\n"
            "          noStMol=RDKit.get_mol(canonical);\n"
            "          if(noStMol&&noStMol.is_valid()){\n"
            "            var canonNS=String(noStMol.get_smiles(JSON.stringify({isomericSmiles:false}))||'').trim();\n"
            "            if(canonNS){matchedEntries=scopeEntries.filter(function(entry){return entry.exact_canonical_nostereo===canonNS||entry.exact_canonical===canonNS;});}\n"
            "          }\n"
            "        }finally{if(noStMol){noStMol.delete();}}\n"
            "      }\n"
            "      _STRUCTURE_MATCH_STATE.lastQuery=canonical;\n"
            "    } else {\n"
            "      var RDKit=await _ensureRDKit();\n"
            "      var queryMol=null;\n"
            "      try {\n"
            "        var srcMol=RDKit.get_mol(inputText);\n"
            "        if(srcMol&&srcMol.is_valid()){\n"
            "          try{\n"
            "            var querySmarts=String(srcMol.get_smarts()||'').trim();\n"
            "            if(!querySmarts){throw new Error('Could not derive a substructure query from the pasted SMILES.');}\n"
            "            queryMol=RDKit.get_qmol(querySmarts);\n"
            "            if(!queryMol||!queryMol.is_valid()){throw new Error('Could not build a valid substructure query from the pasted SMILES.');}\n"
            "            inputText=querySmarts;\n"
            "          }finally{srcMol.delete();}\n"
            "        } else {\n"
            "          if(srcMol){srcMol.delete();}\n"
            "          queryMol=RDKit.get_qmol(inputText);\n"
            "          if(!queryMol||!queryMol.is_valid()){\n"
            "            if(queryMol){queryMol.delete();queryMol=null;}\n"
            "            throw new Error('Paste a valid SMARTS or SMILES string for substructure search.');\n"
            "          }\n"
            "        }\n"
            "        matchedEntries=await _collectMatchesWithProgress(scopeEntries,scope,'substructure',queryMol,'','',RDKit,'search');\n"
            "        _STRUCTURE_MATCH_STATE.lastQuery=inputText;\n"
            "      } finally { if(queryMol){queryMol.delete();} }\n"
            "    }\n"
            "    var matchedScaffolds=new Set();\n"
            "    var matchedMolIds=new Set();\n"
            "    matchedEntries.forEach(function(entry){\n"
            "      matchedScaffolds.add(String(entry.scaffold||''));\n"
            "      matchedMolIds.add(String(entry.mol_id||''));\n"
            "    });\n"
            "    if(matchedScaffolds.size===0){\n"
            "      _STRUCTURE_MATCH_STATE.active=false;\n"
            "      _STRUCTURE_MATCH_STATE.scope=scope;\n"
            "      _applyCentralFilters();\n"
            "      _setStructureSearchStatus('No matches found in '+_scopeLabel(scope)+'. All scaffolds shown.','working');\n"
            "    } else {\n"
            "      _STRUCTURE_MATCH_STATE.active=true;\n"
            "      _STRUCTURE_MATCH_STATE.mode=mode;\n"
            "      _STRUCTURE_MATCH_STATE.scope=scope;\n"
            "      _STRUCTURE_MATCH_STATE.matchedScaffolds=matchedScaffolds;\n"
            "      _STRUCTURE_MATCH_STATE.matchedMolIds=matchedMolIds;\n"
            "      _applyCentralFilters();\n"
            "      var scaffoldCount=_STRUCTURE_MATCH_STATE.matchedScaffolds.size;\n"
            "      var memberCount=_STRUCTURE_MATCH_STATE.matchedMolIds.size;\n"
            "      _setStructureSearchStatus('Matched '+scaffoldCount+' scaffold(s) and '+memberCount+' member(s) in '+_scopeLabel(scope)+'. Matching scaffolds are highlighted with a blue border.','success');\n"
            "    }\n"
            "  } catch (err) {\n"
            "    _STRUCTURE_MATCH_STATE.active=false;\n"
            "    _applyCentralFilters();\n"
            "    _setStructureSearchStatus((err&&err.message)?err.message:'Structure search failed.','error');\n"
            "  }\n"
            "}\n"
            "function _getSelectedHBondResidues(){return Array.prototype.slice.call(document.querySelectorAll('.hbond-residue-filter:checked')).map(function(cb){return String(cb.value||'');}).filter(Boolean);}\n"
            "function _applyCentralFilters(){\n"
            "  _recomputeCentralCards(true);\n"
            "}\n"
            "function _initCentralFilters(){\n"
            "  var qEl=document.getElementById('central-scaffold-search');\n"
            "  var redEl=document.getElementById('filter-red-highlight');\n"
            "  var greenEl=document.getElementById('filter-green-highlight');\n"
            "  var dupEl=document.getElementById('filter-hide-duplicates');\n"
            "  var excludeRunBtn=document.getElementById('exclude-motif-run');\n"
            "  var excludeResetBtn=document.getElementById('exclude-motif-reset');\n"
            "  var excludeInput=document.getElementById('exclude-motif-input');\n"
            "  var runBtn=document.getElementById('structure-search-run');\n"
            "  var resetBtn=document.getElementById('structure-search-reset');\n"
            "  var structureInput=document.getElementById('structure-search-input');\n"
            "  var sizeSel=document.getElementById('central-page-size');\n"
            "  try{\n"
            "    _setSearchScope(_DEFAULT_SEARCH_SCOPE,{announce:false});\n"
            "    document.querySelectorAll('.search-scope-tab').forEach(function(btn){\n"
            "      if(btn.dataset.bound==='1'){return;}\n"
            "      btn.dataset.bound='1';\n"
            "      btn.addEventListener('click',function(){\n"
            "        var scope=String(btn.dataset.searchScope||_DEFAULT_SEARCH_SCOPE);\n"
            "        _setSearchScope(scope,{announce:true});\n"
            "      });\n"
            "    });\n"
            "    if(qEl&&qEl.dataset.bound!=='1'){qEl.dataset.bound='1';qEl.addEventListener('input',_applyCentralFilters);}\n"
            "    if(redEl&&redEl.dataset.bound!=='1'){redEl.dataset.bound='1';redEl.addEventListener('change',_applyCentralFilters);}\n"
            "    // Force rebind red/green even if the old _initCentralFilters already set dataset.bound,\n"
            "    // because the old version bound a different (DOM-only) _applyCentralFilters function.\n"
            "    else if(redEl){redEl.removeEventListener('change',_applyCentralFilters);redEl.addEventListener('change',_applyCentralFilters);}\n"
            "    if(greenEl&&greenEl.dataset.bound!=='1'){greenEl.dataset.bound='1';greenEl.addEventListener('change',_applyCentralFilters);}\n"
            "    else if(greenEl){greenEl.removeEventListener('change',_applyCentralFilters);greenEl.addEventListener('change',_applyCentralFilters);}\n"
            "    if(dupEl&&dupEl.dataset.bound!=='1'){dupEl.dataset.bound='1';dupEl.addEventListener('change',function(){_updateCentralMemberCountLabels();_updateIncludeAllMemberLabels();_applyDuplicateHideToDeepDives();_updateDeepDiveMemberCounts();});}\n"
            "    if(excludeRunBtn&&excludeRunBtn.dataset.bound!=='1'){excludeRunBtn.dataset.bound='1';excludeRunBtn.addEventListener('click',_runExcludeMotif);}\n"
            "    if(excludeResetBtn&&excludeResetBtn.dataset.bound!=='1'){excludeResetBtn.dataset.bound='1';excludeResetBtn.addEventListener('click',function(){_resetExcludeMotif({clearInput:true});});}\n"
            "    if(excludeInput&&excludeInput.dataset.bound!=='1'){excludeInput.dataset.bound='1';excludeInput.addEventListener('keydown',function(evt){if((evt.ctrlKey||evt.metaKey)&&evt.key==='Enter'){evt.preventDefault();_runExcludeMotif();}});}\n"
            "    if(excludeInput&&excludeInput.dataset.warmupBound!=='1'){excludeInput.dataset.warmupBound='1';excludeInput.addEventListener('focus',_warmupStructureSearchAssets,{once:true});}\n"
            "    if(runBtn&&runBtn.dataset.bound!=='1'){runBtn.dataset.bound='1';runBtn.addEventListener('click',_runStructureSearch);}\n"
            "    if(runBtn&&runBtn.dataset.warmupBound!=='1'){runBtn.dataset.warmupBound='1';runBtn.addEventListener('mouseenter',_warmupStructureSearchAssets,{once:true});runBtn.addEventListener('focus',_warmupStructureSearchAssets,{once:true});}\n"
            "    if(resetBtn&&resetBtn.dataset.bound!=='1'){resetBtn.dataset.bound='1';resetBtn.addEventListener('click',function(){_resetStructureSearch();});}\n"
            "    if(structureInput&&structureInput.dataset.bound!=='1'){structureInput.dataset.bound='1';structureInput.addEventListener('keydown',function(evt){if((evt.ctrlKey||evt.metaKey)&&evt.key==='Enter'){evt.preventDefault();_runStructureSearch();}});}\n"
            "    if(structureInput&&structureInput.dataset.warmupBound!=='1'){structureInput.dataset.warmupBound='1';structureInput.addEventListener('focus',_warmupStructureSearchAssets,{once:true});}\n"
            "    document.querySelectorAll('.hbond-residue-filter').forEach(function(cb){if(cb.dataset.bound!=='1'){cb.dataset.bound='1';cb.addEventListener('change',_applyCentralFilters);}});\n"
            "    if(sizeSel&&sizeSel.dataset.bound!=='1'){sizeSel.dataset.bound='1';sizeSel.addEventListener('change',function(){_CENTRAL_RENDER_STATE.page=0;_renderCentralPage();});}\n"
            "    function _bindPagerBtn(id,fn){var btn=document.getElementById(id);if(btn&&btn.dataset.bound!=='1'){btn.dataset.bound='1';btn.addEventListener('click',fn);}}\n"
            "    _bindPagerBtn('central-page-first',function(){_CENTRAL_RENDER_STATE.page=0;_renderCentralPage();});\n"
            "    _bindPagerBtn('central-page-prev',function(){if(_CENTRAL_RENDER_STATE.page>0){_CENTRAL_RENDER_STATE.page-=1;_renderCentralPage();}});\n"
            "    _bindPagerBtn('central-page-next',function(){_CENTRAL_RENDER_STATE.page+=1;_renderCentralPage();});\n"
            "    _bindPagerBtn('central-page-last',function(){_CENTRAL_RENDER_STATE.page=Math.max(0,_CENTRAL_RENDER_STATE.lastTotalPages-1);_renderCentralPage();});\n"
            "    _bindPagerBtn('central-page-first-bottom',function(){_CENTRAL_RENDER_STATE.page=0;_renderCentralPage();});\n"
            "    _bindPagerBtn('central-page-prev-bottom',function(){if(_CENTRAL_RENDER_STATE.page>0){_CENTRAL_RENDER_STATE.page-=1;_renderCentralPage();}});\n"
            "    _bindPagerBtn('central-page-next-bottom',function(){_CENTRAL_RENDER_STATE.page+=1;_renderCentralPage();});\n"
            "    _bindPagerBtn('central-page-last-bottom',function(){_CENTRAL_RENDER_STATE.page=Math.max(0,_CENTRAL_RENDER_STATE.lastTotalPages-1);_renderCentralPage();});\n"
            "    _CENTRAL_RENDER_STATE.filteredScaffolds=_getCentralCards().map(function(entry){return String(entry.scaffold||'');});\n"
            "    _updateHighlightListing();\n"
            "    _applyCentralFilters();\n"
            "  }catch(err){\n"
            "    console.error('Central init failed:',err);\n"
            "    if(typeof _setStructureSearchStatus==='function'){_setStructureSearchStatus('Central filters failed to initialize. See browser console for details.','error');}\n"
            "  }\n"
            "}\n"
            "// ---- Properties panel JS ----\n"
            "function _scaffoldPassesPropFilter(scaffoldName){\n"
            "  if(typeof _PROP_FILTER_STATE==='undefined'){return true;}\n"
            "  var activeFilters=Object.keys(_PROP_FILTER_STATE);\n"
            "  if(!activeFilters.length){return true;}\n"
            "  var members=_getScaffoldMolMap()[scaffoldName]||[];\n"
            "  if(!members.length){return true;}\n"
            "  return members.some(function(molId){\n"
            "    var props=_getMolPropsData()[molId];\n"
            "    if(!props){return false;}\n"
            "    return activeFilters.every(function(p){\n"
            "      var pIdx=_PROP_NAMES.indexOf(p);\n"
            "      if(pIdx<0){return true;}\n"
            "      var range=_PROP_FILTER_STATE[p];\n"
            "      var val=props[pIdx];\n"
            "      return val!==null&&val!==undefined&&val>=range.min&&val<=range.max;\n"
            "    });\n"
            "  });\n"
            "}\n"
            "function _applyPropFilterToDeepDives(){\n"
            "  if(typeof _PROP_FILTER_STATE==='undefined'){return;}\n"
            "  var activeFilters=Object.keys(_PROP_FILTER_STATE);\n"
            "  document.querySelectorAll('#deep-dive-shell .moltile').forEach(function(tile){\n"
            "    var pass=true;\n"
            "    if(activeFilters.length){\n"
            "      var molId=String(tile.dataset.molId||'');\n"
            "      var props=_getMolPropsData()[molId]||null;\n"
            "      if(props){\n"
            "        pass=activeFilters.every(function(p){\n"
            "          var pIdx=_PROP_NAMES.indexOf(p);\n"
            "          if(pIdx<0){return true;}\n"
            "          var range=_PROP_FILTER_STATE[p];\n"
            "          var val=props[pIdx];\n"
            "          return val!==null&&val!==undefined&&val>=range.min&&val<=range.max;\n"
            "        });\n"
            "      }\n"
            "    }\n"
            "    tile.dataset.propHidden=pass?'0':'1';\n"
            "  });\n"
            "  _applyDuplicateHideToDeepDives();\n"
            "  _updateDeepDiveMemberCounts();\n"
            "}\n"
            "function _showPropTab(tabName){\n"
            "  document.querySelectorAll('.prop-tab-content').forEach(function(el){el.style.display='none';});\n"
            "  document.querySelectorAll('.prop-tab-btn').forEach(function(el){el.classList.remove('active');});\n"
            "  var c=document.getElementById('prop-tab-'+tabName);\n"
            "  if(c){c.style.display='';}\n"
            "  document.querySelectorAll('.prop-tab-btn[data-tab=\"'+tabName+'\"]').forEach(function(el){el.classList.add('active');});\n"
            "  if(tabName==='hist'){_updatePropHistogram();}\n"
            "  if(tabName==='box'){_updatePropBoxPlot();}\n"
            "}\n"
            "function _ensurePlotly(){\n"
            "  if(!_plotlyReadyPromise){\n"
            "    if(window.Plotly){_plotlyReadyPromise=Promise.resolve(window.Plotly);}\n"
            "    else{\n"
            "      _plotlyReadyPromise=new Promise(function(resolve,reject){\n"
            "        var s=document.createElement('script');\n"
            "        s.src='https://cdn.plot.ly/plotly-2.26.0.min.js';\n"
            "        s.onload=function(){resolve(window.Plotly);};\n"
            "        s.onerror=function(){\n"
            "          var el=document.getElementById('prop-plotly-status');\n"
            "          if(el){el.style.display='';}\n"
            "          reject(new Error('Plotly load failed'));\n"
            "        };\n"
            "        document.head.appendChild(s);\n"
            "      });\n"
            "    }\n"
            "  }\n"
            "  return _plotlyReadyPromise;\n"
            "}\n"
            "function _updatePropHistogram(){\n"
            "  if(typeof _PROP_NAMES==='undefined'){return;}\n"
            "  var sel=document.getElementById('prop-hist-select');\n"
            "  if(!sel){return;}\n"
            "  var propName=sel.value;\n"
            "  var propIdx=_PROP_NAMES.indexOf(propName);\n"
            "  if(propIdx<0){return;}\n"
            "  var vals=[];\n"
            "  Object.values(_getMolPropsData()).forEach(function(arr){if(arr&&arr[propIdx]!==null&&arr[propIdx]!==undefined){vals.push(arr[propIdx]);}});\n"
            "  var label=(_PROP_LABELS&&_PROP_LABELS[propName])||propName;\n"
            "  var activeFilter=_PROP_FILTER_STATE[propName];\n"
            "  var shapes=[];\n"
            "  if(activeFilter){\n"
            "    shapes=[{type:'line',x0:activeFilter.min,x1:activeFilter.min,y0:0,y1:1,yref:'paper',line:{color:'#e33',dash:'dot',width:2}},\n"
            "             {type:'line',x0:activeFilter.max,x1:activeFilter.max,y0:0,y1:1,yref:'paper',line:{color:'#e33',dash:'dot',width:2}}];\n"
            "  }\n"
            "  var minEl=document.getElementById('prop-range-min');\n"
            "  var maxEl=document.getElementById('prop-range-max');\n"
            "  if(minEl&&maxEl){\n"
            "    if(activeFilter){minEl.value=activeFilter.min;maxEl.value=activeFilter.max;}\n"
            "    else if(vals.length){\n"
            "      var gMin=Math.min.apply(null,vals);var gMax=Math.max.apply(null,vals);\n"
            "      minEl.value=Math.round(gMin*1000)/1000;maxEl.value=Math.round(gMax*1000)/1000;\n"
            "    }\n"
            "  }\n"
            "  var chartEl=document.getElementById('prop-histogram-chart');\n"
            "  if(!chartEl){return;}\n"
            "  _ensurePlotly().then(function(Plotly){\n"
            "    Plotly.react(chartEl,[{type:'histogram',x:vals,nbinsx:50,name:'molecules',marker:{color:'rgba(80,150,220,0.6)'}}],\n"
            "      {margin:{t:10,b:50,l:50,r:15},height:230,xaxis:{title:label},yaxis:{title:'Count'},bargap:0.02,shapes:shapes},\n"
            "      {displayModeBar:false,responsive:true});\n"
            "  }).catch(function(){});\n"
            "}\n"
            "function _updatePropBoxPlot(){\n"
            "  if(typeof _PROP_NAMES==='undefined'){return;}\n"
            "  var sel=document.getElementById('prop-box-select');\n"
            "  if(!sel){return;}\n"
            "  var propName=sel.value;\n"
            "  var propIdx=_PROP_NAMES.indexOf(propName);\n"
            "  if(propIdx<0){return;}\n"
            "  var filteredOnly=document.getElementById('prop-box-filtered');\n"
            "  var useFiltered=filteredOnly&&filteredOnly.checked;\n"
            "  var activeFilters=Object.keys(_PROP_FILTER_STATE||{});\n"
            "  var vals=[];\n"
            "  var propsById=_getMolPropsData();\n"
            "  Object.keys(propsById).forEach(function(mid){\n"
            "    var arr=propsById[mid];\n"
            "    if(!arr){return;}\n"
            "    if(useFiltered&&activeFilters.length){\n"
            "      var pass=activeFilters.every(function(p){\n"
            "        var pIdx2=_PROP_NAMES.indexOf(p);\n"
            "        if(pIdx2<0){return true;}\n"
            "        var range=_PROP_FILTER_STATE[p];\n"
            "        var v=arr[pIdx2];\n"
            "        return v!==null&&v!==undefined&&v>=range.min&&v<=range.max;\n"
            "      });\n"
            "      if(!pass){return;}\n"
            "    }\n"
            "    var v0=arr[propIdx];\n"
            "    if(v0!==null&&v0!==undefined){vals.push(v0);}\n"
            "  });\n"
            "  var chartEl=document.getElementById('prop-boxplot-chart');\n"
            "  if(!chartEl){return;}\n"
            "  var label=(_PROP_LABELS&&_PROP_LABELS[propName])||propName;\n"
            "  _ensurePlotly().then(function(Plotly){\n"
            "    Plotly.react(chartEl,[{type:'box',name:label,y:vals,boxpoints:'outliers',jitter:0.28,pointpos:0,\n"
            "      marker:{size:5,color:'rgba(31,79,122,0.55)',line:{color:'rgba(19,46,74,0.85)',width:0.6}},\n"
            "      line:{color:'rgba(16,80,110,0.95)',width:2},fillcolor:'rgba(86,168,191,0.32)',whiskerwidth:0.7}],\n"
            "      {margin:{t:12,b:58,l:66,r:24},height:300,paper_bgcolor:'#ffffff',plot_bgcolor:'#fbfdff',showlegend:false,\n"
            "       xaxis:{title:'Property',tickfont:{size:12,color:'#264057'}},\n"
            "       yaxis:{title:label,gridcolor:'rgba(38,64,87,0.12)',zeroline:false,tickfont:{size:12,color:'#264057'}},\n"
            "       annotations:[{xref:'paper',yref:'paper',x:0,y:1.12,showarrow:false,text:'N = '+vals.length,font:{size:12,color:'#4a6278'}}]},\n"
            "      {displayModeBar:false,responsive:true});\n"
            "  }).catch(function(){});\n"
            "}\n"
            "function _applyPropRangeFilter(){\n"
            "  if(typeof _PROP_FILTER_STATE==='undefined'){return;}\n"
            "  var sel=document.getElementById('prop-hist-select');\n"
            "  if(!sel){return;}\n"
            "  var propName=sel.value;\n"
            "  var minEl=document.getElementById('prop-range-min');\n"
            "  var maxEl=document.getElementById('prop-range-max');\n"
            "  if(!minEl||!maxEl){return;}\n"
            "  var minVal=parseFloat(minEl.value);var maxVal=parseFloat(maxEl.value);\n"
            "  if(isNaN(minVal)||isNaN(maxVal)){return;}\n"
            "  if(minVal>maxVal){var t=minVal;minVal=maxVal;maxVal=t;}\n"
            "  _PROP_FILTER_STATE[propName]={min:minVal,max:maxVal};\n"
            "  _updateActiveFilterChips();\n"
            "  _updatePropHistogram();\n"
            "  _updatePropBoxPlot();\n"
            "  _applyCentralFilters();\n"
            "  _applyPropFilterToDeepDives();\n"
            "}\n"
            "function _resetCurrentPropFilter(){\n"
            "  if(typeof _PROP_FILTER_STATE==='undefined'){return;}\n"
            "  var sel=document.getElementById('prop-hist-select');\n"
            "  if(!sel){return;}\n"
            "  delete _PROP_FILTER_STATE[sel.value];\n"
            "  _updateActiveFilterChips();\n"
            "  _updatePropHistogram();\n"
            "  _updatePropBoxPlot();\n"
            "  _applyCentralFilters();\n"
            "  _applyPropFilterToDeepDives();\n"
            "}\n"
            "function _clearAllPropFilters(){\n"
            "  if(typeof _PROP_FILTER_STATE==='undefined'){return;}\n"
            "  Object.keys(_PROP_FILTER_STATE).forEach(function(k){delete _PROP_FILTER_STATE[k];});\n"
            "  _updateActiveFilterChips();\n"
            "  _updatePropHistogram();\n"
            "  _updatePropBoxPlot();\n"
            "  _applyCentralFilters();\n"
            "  _applyPropFilterToDeepDives();\n"
            "}\n"
            "function _removePropFilter(propName){\n"
            "  if(typeof _PROP_FILTER_STATE==='undefined'){return;}\n"
            "  delete _PROP_FILTER_STATE[propName];\n"
            "  _updateActiveFilterChips();\n"
            "  var sel=document.getElementById('prop-hist-select');\n"
            "  if(sel&&sel.value===propName){_updatePropHistogram();}\n"
            "  _updatePropBoxPlot();\n"
            "  _applyCentralFilters();\n"
            "  _applyPropFilterToDeepDives();\n"
            "}\n"
            "function _updateActiveFilterChips(){\n"
            "  if(typeof _PROP_FILTER_STATE==='undefined'){return;}\n"
            "  var chipsEl=document.getElementById('prop-filter-chips');\n"
            "  var bannerEl=document.getElementById('prop-active-filters');\n"
            "  var keys=Object.keys(_PROP_FILTER_STATE);\n"
            "  if(bannerEl){bannerEl.style.display=keys.length?'':'none';}\n"
            "  if(!chipsEl){return;}\n"
            "  chipsEl.innerHTML=keys.map(function(p){\n"
            "    var range=_PROP_FILTER_STATE[p];\n"
            "    var lbl=(_PROP_LABELS&&_PROP_LABELS[p])||p;\n"
            "    var mn=Math.round(range.min*10000)/10000;\n"
            "    var mx=Math.round(range.max*10000)/10000;\n"
            "    return '<span class=\"prop-filter-chip-tag\">'+lbl+': '+mn+'\\u2013'+mx\n"
            "      +'<button type=\"button\" onclick=\"_removePropFilter(\\''+p+'\\')\">&times;</button></span>';\n"
            "  }).join('');\n"
            "}\n"
            "var _corrHoverSvgCache={};\n"
            "function _escapeHtml(text){\n"
            "  return String(text===null||text===undefined?'':text)\n"
            "    .replace(/&/g,'&amp;')\n"
            "    .replace(/</g,'&lt;')\n"
            "    .replace(/>/g,'&gt;')\n"
            "    .replace(/\"/g,'&quot;')\n"
            "    .replace(/'/g,'&#39;');\n"
            "}\n"
            "function _formatCorrValue(value){\n"
            "  if(value===null||value===undefined||value===''){return '\u2014';}\n"
            "  var num=Number(value);\n"
            "  if(isFinite(num)){return String(Math.round(num*10000)/10000);}\n"
            "  return String(value);\n"
            "}\n"
            "function _getCorrPointPayload(pointData){\n"
            "  var custom=Array.isArray(pointData&&pointData.customdata)?pointData.customdata:[];\n"
            "  return {\n"
            "    molId:String(custom[0]!==undefined?custom[0]:(pointData&&pointData.text)||''),\n"
            "    smiles:String(custom[1]||''),\n"
            "    x:(pointData&&pointData.x),\n"
            "    y:(pointData&&pointData.y)\n"
            "  };\n"
            "}\n"
            "async function _getCorrStructureSvg(smiles){\n"
            "  var key=String(smiles||'').trim();\n"
            "  if(!key){return ''; }\n"
            "  if(!_corrHoverSvgCache[key]){\n"
            "    _corrHoverSvgCache[key]=_ensureRDKit().then(function(RDKit){\n"
            "      var mol=null;\n"
            "      try{\n"
            "        mol=RDKit.get_mol(key);\n"
            "        if(!mol){return ''; }\n"
            "        return mol.get_svg()||'';\n"
            "      }catch(err){\n"
            "        return '';\n"
            "      }finally{\n"
            "        if(mol){try{mol.delete();}catch(_err){}}\n"
            "      }\n"
            "    }).catch(function(){return '';});\n"
            "  }\n"
            "  return _corrHoverSvgCache[key];\n"
            "}\n"
            "function _buildCorrCardHtml(payload,svgMarkup,xLabel,yLabel){\n"
            "  var structureHtml=svgMarkup\n"
            "    ? '<div class=\"prop-corr-structure\">'+svgMarkup+'</div>'\n"
            "    : '<div class=\"prop-corr-structure smalltxt\">2D structure unavailable</div>';\n"
            "  return structureHtml\n"
            "    +'<div class=\"prop-corr-meta\"><div><b>'+_escapeHtml(payload.molId)+'</b></div>'\n"
            "    +'<div>'+_escapeHtml(xLabel)+': '+_escapeHtml(_formatCorrValue(payload.x))+'</div>'\n"
            "    +'<div>'+_escapeHtml(yLabel)+': '+_escapeHtml(_formatCorrValue(payload.y))+'</div></div>';\n"
            "}\n"
            "function _hideCorrHover(){\n"
            "  var hoverEl=document.getElementById('prop-corr-hover');\n"
            "  if(!hoverEl){return;}\n"
            "  hoverEl.style.display='none';\n"
            "  hoverEl.innerHTML='';\n"
            "}\n"
            "function _positionCorrHover(hoverEl,evt){\n"
            "  if(!hoverEl||!evt){return;}\n"
            "  var left=evt.clientX+16;\n"
            "  var top=evt.clientY+16;\n"
            "  var maxLeft=Math.max(12,window.innerWidth-hoverEl.offsetWidth-12);\n"
            "  var maxTop=Math.max(12,window.innerHeight-hoverEl.offsetHeight-12);\n"
            "  hoverEl.style.left=Math.max(12,Math.min(left,maxLeft))+'px';\n"
            "  hoverEl.style.top=Math.max(12,Math.min(top,maxTop))+'px';\n"
            "}\n"
            "async function _showCorrHover(pointData,xLabel,yLabel,evt){\n"
            "  var hoverEl=document.getElementById('prop-corr-hover');\n"
            "  if(!hoverEl){return;}\n"
            "  var payload=_getCorrPointPayload(pointData);\n"
            "  hoverEl.style.display='block';\n"
            "  hoverEl.innerHTML=_buildCorrCardHtml(payload,'',xLabel,yLabel);\n"
            "  _positionCorrHover(hoverEl,evt);\n"
            "  var token=String(payload.molId)+'|'+String(payload.x)+'|'+String(payload.y);\n"
            "  hoverEl.dataset.token=token;\n"
            "  var svgMarkup=await _getCorrStructureSvg(payload.smiles||((typeof _MOL_SMILES_BY_ID!=='undefined'&&_MOL_SMILES_BY_ID[payload.molId])||''));\n"
            "  if(hoverEl.dataset.token!==token){return;}\n"
            "  hoverEl.innerHTML=_buildCorrCardHtml(payload,svgMarkup,xLabel,yLabel);\n"
            "  _positionCorrHover(hoverEl,evt);\n"
            "}\n"
            "async function _pinCorrPoint(pointData,xLabel,yLabel){\n"
            "  var detailEl=document.getElementById('prop-corr-click-detail');\n"
            "  if(!detailEl){return;}\n"
            "  var payload=_getCorrPointPayload(pointData);\n"
            "  detailEl.classList.remove('empty');\n"
            "  detailEl.innerHTML=_buildCorrCardHtml(payload,'',xLabel,yLabel);\n"
            "  var svgMarkup=await _getCorrStructureSvg(payload.smiles||((typeof _MOL_SMILES_BY_ID!=='undefined'&&_MOL_SMILES_BY_ID[payload.molId])||''));\n"
            "  detailEl.innerHTML=_buildCorrCardHtml(payload,svgMarkup,xLabel,yLabel);\n"
            "}\n"
            "function _plotCorrelation(){\n"
            "  if(typeof _PROP_NAMES==='undefined'){return;}\n"
            "  var xSel=document.getElementById('prop-corr-x');\n"
            "  var ySel=document.getElementById('prop-corr-y');\n"
            "  if(!xSel||!ySel){return;}\n"
            "  var xProp=xSel.value;var yProp=ySel.value;\n"
            "  var xIdx=_PROP_NAMES.indexOf(xProp);var yIdx=_PROP_NAMES.indexOf(yProp);\n"
            "  if(xIdx<0||yIdx<0){return;}\n"
            "  var filteredOnly=document.getElementById('prop-corr-filtered');\n"
            "  var useFiltered=filteredOnly&&filteredOnly.checked;\n"
            "  var xs=[],ys=[],texts=[],customdata=[];\n"
            "  var activeFilters=Object.keys(_PROP_FILTER_STATE);\n"
            "  var propsById=_getMolPropsData();\n"
            "  Object.keys(propsById).forEach(function(mid){\n"
            "    var arr=propsById[mid];\n"
            "    if(!arr){return;}\n"
            "    if(useFiltered&&activeFilters.length){\n"
            "      var pass=activeFilters.every(function(p){\n"
            "        var pIdx2=_PROP_NAMES.indexOf(p);if(pIdx2<0){return true;}\n"
            "        var range=_PROP_FILTER_STATE[p];var v=arr[pIdx2];\n"
            "        return v!==null&&v!==undefined&&v>=range.min&&v<=range.max;\n"
            "      });\n"
            "      if(!pass){return;}\n"
            "    }\n"
            "    var xv=arr[xIdx];var yv=arr[yIdx];\n"
            "    if(xv===null||xv===undefined||yv===null||yv===undefined){return;}\n"
            "    xs.push(xv);ys.push(yv);texts.push(mid);customdata.push([mid,((typeof _MOL_SMILES_BY_ID!=='undefined'&&_MOL_SMILES_BY_ID[mid])||'')]);\n"
            "  });\n"
            "  var xLabel=(_PROP_LABELS&&_PROP_LABELS[xProp])||xProp;\n"
            "  var yLabel=(_PROP_LABELS&&_PROP_LABELS[yProp])||yProp;\n"
            "  var chartEl=document.getElementById('prop-corr-chart');\n"
            "  if(!chartEl){return;}\n"
            "  _hideCorrHover();\n"
            "  _ensurePlotly().then(function(Plotly){\n"
            "    Plotly.react(chartEl,\n"
            "      [\n"
            "        {type:'histogram2dcontour',name:'density',x:xs,y:ys,xaxis:'x',yaxis:'y',hoverinfo:'skip',showscale:true,ncontours:16,\n"
            "         colorscale:[[0,'rgba(44,123,182,0.05)'],[0.2,'rgba(67,162,202,0.18)'],[0.45,'rgba(127,205,187,0.35)'],[0.7,'rgba(253,174,97,0.55)'],[1,'rgba(215,25,28,0.78)']],\n"
            "         contours:{coloring:'fill',showlines:true},line:{color:'rgba(150,27,33,0.62)',width:0.8},\n"
            "         colorbar:{title:'Density',titleside:'right',len:0.78,thickness:11,x:1.03,y:0.41}},\n"
            "        {type:'scattergl',name:'points',mode:'markers',x:xs,y:ys,text:texts,customdata:customdata,xaxis:'x',yaxis:'y',hoverinfo:'none',\n"
            "         marker:{size:5,opacity:0.42,color:'rgba(24,96,180,0.55)'}},\n"
            "        {type:'histogram',x:xs,xaxis:'x2',yaxis:'y2',nbinsx:50,hoverinfo:'skip',marker:{color:'rgba(24,96,180,0.28)'},showlegend:false},\n"
            "        {type:'histogram',y:ys,orientation:'h',xaxis:'x3',yaxis:'y3',nbinsy:50,hoverinfo:'skip',marker:{color:'rgba(24,96,180,0.28)'},showlegend:false}\n"
            "      ],\n"
            "      {margin:{t:24,b:60,l:60,r:24},height:520,showlegend:false,bargap:0.03,hovermode:'closest',dragmode:'zoom',\n"
            "       paper_bgcolor:'#ffffff',plot_bgcolor:'#ffffff',\n"
            "       xaxis:{domain:[0,0.84],title:xLabel,zeroline:false},\n"
            "       yaxis:{domain:[0,0.84],title:yLabel,zeroline:false},\n"
            "       xaxis2:{domain:[0,0.84],anchor:'y2',showgrid:false,zeroline:false,showticklabels:false},\n"
            "       yaxis2:{domain:[0.86,1],anchor:'x2',showgrid:false,zeroline:false,showticklabels:false},\n"
            "       xaxis3:{domain:[0.86,1],anchor:'y3',showgrid:false,zeroline:false,showticklabels:false},\n"
            "       yaxis3:{domain:[0,0.84],anchor:'x3',showgrid:false,zeroline:false,showticklabels:false}},\n"
            "      {displayModeBar:true,responsive:true,scrollZoom:true,doubleClick:'reset',modeBarButtonsToAdd:['select2d','lasso2d']});\n"
            "    if(typeof chartEl.removeAllListeners==='function'){\n"
            "      chartEl.removeAllListeners('plotly_hover');\n"
            "      chartEl.removeAllListeners('plotly_unhover');\n"
            "      chartEl.removeAllListeners('plotly_click');\n"
            "    }\n"
            "    chartEl.on('plotly_hover',function(evt){\n"
            "      var point=evt&&evt.points&&evt.points.length?evt.points[0]:null;\n"
            "      if(!point||!point.data||point.data.name!=='points'){return;}\n"
            "      _showCorrHover(point,xLabel,yLabel,evt.event||null);\n"
            "    });\n"
            "    chartEl.on('plotly_unhover',function(){_hideCorrHover();});\n"
            "    chartEl.on('plotly_click',function(evt){\n"
            "      var point=evt&&evt.points&&evt.points.length?evt.points[0]:null;\n"
            "      if(!point||!point.data||point.data.name!=='points'){return;}\n"
            "      _pinCorrPoint(point,xLabel,yLabel);\n"
            "    });\n"
            "  }).catch(function(){});\n"
            "}\n"
            "function _initPropPanel(){\n"
            "  if(typeof _PROP_NAMES==='undefined'){return;}\n"
            "  var histSel=document.getElementById('prop-hist-select');\n"
            "  var corrX=document.getElementById('prop-corr-x');\n"
            "  var corrY=document.getElementById('prop-corr-y');\n"
            "  // Correlation selects are already populated in HTML; histogram select too.\n"
            "  // Set default correlation Y to second property.\n"
            "  if(corrY&&_PROP_NAMES.length>1){corrY.value=_PROP_NAMES[1];}\n"
            "  // Defer histogram/correlation plotting until user opens the tab or clicks Plot.\n"
            "}\n"
            "_initCentralFilters();\n"
            "_initPropPanel();\n"
            "</script>"
        )
        fh.write("</div></body></html>")