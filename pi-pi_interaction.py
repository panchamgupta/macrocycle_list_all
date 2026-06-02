#!/usr/bin/env python3
"""Protein-ligand interaction calculator with rigorous pi-pi detection.

This module preserves the viewer interaction schema from sdf-viewer-offline
while upgrading aromatic interaction detection from atom-distance heuristics
to ring centroid and orientation criteria.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem


INTERACTION_TYPES = ("hbond", "salt", "aromatic", "hydrophobic", "halogen")
AROMATIC_PROTEIN_RESIDUES = {"PHE", "TYR", "TRP", "HIS"}
HALOGEN_ELEMS = {"F", "CL", "BR", "I"}


# Side-chain aromatic rings used to compute residue ring centroids/normals.
PROTEIN_RING_ATOM_SETS = {
    "PHE": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
    "TYR": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
    "HIS": [["CG", "ND1", "CD2", "CE1", "NE2"]],
    "TRP": [
        ["CG", "CD1", "NE1", "CE2", "CD2"],
        ["CD2", "CE2", "CE3", "CD3", "CZ3", "CH2", "CZ2"],
    ],
}


@dataclass(frozen=True)
class AtomRecord:
    serial: int
    name: str
    resn: str
    chain: str
    resi: int
    elem: str
    x: float
    y: float
    z: float

    @property
    def coord(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass(frozen=True)
class RingRecord:
    residue_key: str
    atom_serials: Tuple[int, ...]
    coords: np.ndarray
    centroid: np.ndarray
    normal: np.ndarray
    subtype: str = ""


def _safe_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except Exception:
        return None


def _infer_elem(atom_name: str) -> str:
    trimmed = atom_name.strip().upper()
    if not trimmed:
        return ""
    two = trimmed[:2]
    if two in HALOGEN_ELEMS:
        return two
    return trimmed[0]


def parse_pdb_atoms(pdb_path: Path) -> List[AtomRecord]:
    atoms: List[AtomRecord] = []
    with pdb_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            serial = int(line[6:11].strip() or 0)
            name = line[12:16].strip().upper()
            resn = line[17:20].strip().upper()
            chain = line[21:22].strip()
            resi = int(line[22:26].strip() or 0)
            x = _safe_float(line[30:38].strip())
            y = _safe_float(line[38:46].strip())
            z = _safe_float(line[46:54].strip())
            if x is None or y is None or z is None:
                continue
            elem = line[76:78].strip().upper() or _infer_elem(name)
            atoms.append(
                AtomRecord(
                    serial=serial,
                    name=name,
                    resn=resn,
                    chain=chain,
                    resi=resi,
                    elem=elem,
                    x=x,
                    y=y,
                    z=z,
                )
            )
    return atoms


def _fit_plane_normal(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    n = np.linalg.norm(normal)
    if n <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return normal / n


def _angle_between_normals_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    d = float(np.dot(n1, n2))
    d = max(-1.0, min(1.0, d))
    angle = math.degrees(math.acos(abs(d)))
    return angle


def _empty_payload() -> Dict[str, object]:
    return {
        "proteinAtomSerials": set(),
        "ligandAtomSerials": set(),
        "residueKeys": set(),
        "pairs": [],
    }


def _distance2(a: AtomRecord, b: AtomRecord) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return dx * dx + dy * dy + dz * dz


def _is_hbond_atom(atom: AtomRecord) -> bool:
    return atom.elem in {"N", "O", "S"}


def _is_hydrophobic_atom(atom: AtomRecord) -> bool:
    return atom.elem in {"C", "S"}


def _is_acidic_protein_atom(atom: AtomRecord) -> bool:
    if atom.resn == "ASP":
        return atom.name in {"OD1", "OD2"}
    if atom.resn == "GLU":
        return atom.name in {"OE1", "OE2"}
    return False


def _is_basic_protein_atom(atom: AtomRecord) -> bool:
    if atom.resn == "LYS":
        return atom.name == "NZ"
    if atom.resn == "ARG":
        return atom.name in {"NE", "NH1", "NH2"}
    if atom.resn == "HIS":
        return atom.name in {"ND1", "NE2"}
    return False


def _is_acidic_ligand_atom(atom: AtomRecord) -> bool:
    return atom.elem in {"O", "S"}


def _is_basic_ligand_atom(atom: AtomRecord) -> bool:
    return atom.elem == "N"


def _add_pair_hit(by_type: Dict[str, Dict[str, object]], itype: str, pa: AtomRecord, la: AtomRecord) -> None:
    payload = by_type[itype]
    payload["proteinAtomSerials"].add(pa.serial)
    payload["ligandAtomSerials"].add(la.serial)
    payload["residueKeys"].add(f"{pa.chain}|{pa.resi}|{pa.resn}")
    payload["pairs"].append(
        {
            "proteinSerial": pa.serial,
            "ligandSerial": la.serial,
            "p": {"x": pa.x, "y": pa.y, "z": pa.z},
            "l": {"x": la.x, "y": la.y, "z": la.z},
        }
    )


def build_protein_aromatic_rings(protein_atoms: Sequence[AtomRecord]) -> List[RingRecord]:
    residue_atoms: Dict[Tuple[str, int, str], Dict[str, AtomRecord]] = defaultdict(dict)
    for atom in protein_atoms:
        residue_atoms[(atom.chain, atom.resi, atom.resn)][atom.name] = atom

    rings: List[RingRecord] = []
    for (chain, resi, resn), name_map in residue_atoms.items():
        if resn not in AROMATIC_PROTEIN_RESIDUES:
            continue
        for atom_names in PROTEIN_RING_ATOM_SETS.get(resn, []):
            picked = [name_map.get(name) for name in atom_names]
            if any(a is None for a in picked):
                continue
            ring_atoms = [a for a in picked if a is not None]
            coords = np.vstack([a.coord for a in ring_atoms])
            centroid = coords.mean(axis=0)
            normal = _fit_plane_normal(coords)
            rings.append(
                RingRecord(
                    residue_key=f"{chain}|{resi}|{resn}",
                    atom_serials=tuple(a.serial for a in ring_atoms),
                    coords=coords,
                    centroid=centroid,
                    normal=normal,
                )
            )
    return rings


def ligand_atoms_and_rings(mol: Chem.Mol) -> Tuple[List[AtomRecord], List[RingRecord]]:
    conf = mol.GetConformer()
    atoms: List[AtomRecord] = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        atoms.append(
            AtomRecord(
                serial=idx,
                name=atom.GetSymbol().upper(),
                resn="LIG",
                chain="",
                resi=1,
                elem=atom.GetSymbol().upper(),
                x=float(pos.x),
                y=float(pos.y),
                z=float(pos.z),
            )
        )

    rings: List[RingRecord] = []
    ring_info = mol.GetRingInfo()
    for ring_atom_indices in ring_info.AtomRings():
        aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_atom_indices)
        if not aromatic:
            continue
        ring_atoms = [atoms[i] for i in ring_atom_indices]
        coords = np.vstack([a.coord for a in ring_atoms])
        centroid = coords.mean(axis=0)
        normal = _fit_plane_normal(coords)
        rings.append(
            RingRecord(
                residue_key="|1|LIG",
                atom_serials=tuple(a.serial for a in ring_atoms),
                coords=coords,
                centroid=centroid,
                normal=normal,
            )
        )

    return atoms, rings


def _nearest_pair(
    protein_atoms: Sequence[AtomRecord],
    ligand_atoms: Sequence[AtomRecord],
    protein_serials: Sequence[int],
    ligand_serials: Sequence[int],
) -> Tuple[AtomRecord, AtomRecord]:
    p_map = {a.serial: a for a in protein_atoms}
    l_map = {a.serial: a for a in ligand_atoms}
    best = None
    best_d2 = float("inf")
    for ps in protein_serials:
        pa = p_map.get(ps)
        if pa is None:
            continue
        for ls in ligand_serials:
            la = l_map.get(ls)
            if la is None:
                continue
            d2 = _distance2(pa, la)
            if d2 < best_d2:
                best = (pa, la)
                best_d2 = d2
    if best is None:
        raise ValueError("No nearest atom pair could be determined for aromatic ring hit")
    return best


def classify_interactions(
    protein_atoms: Sequence[AtomRecord],
    ligand_atoms: Sequence[AtomRecord],
    protein_rings: Sequence[RingRecord],
    ligand_rings: Sequence[RingRecord],
    *,
    include_non_pi: bool = True,
    centroid_cutoff: float = 6.0,
    parallel_angle_max: float = 30.0,
    tshape_angle_min: float = 60.0,
    tshape_angle_max: float = 120.0,
) -> Dict[str, Dict[str, object]]:
    by_type = {itype: _empty_payload() for itype in INTERACTION_TYPES}

    if include_non_pi:
        for pa in protein_atoms:
            for la in ligand_atoms:
                d2 = _distance2(pa, la)
                if d2 > 25.0:
                    continue

                # Keep original viewer priority order for non-pi contacts.
                if d2 <= 16.0:
                    salt_like = (_is_acidic_protein_atom(pa) and _is_basic_ligand_atom(la)) or (
                        _is_basic_protein_atom(pa) and _is_acidic_ligand_atom(la)
                    )
                    if salt_like:
                        _add_pair_hit(by_type, "salt", pa, la)
                        continue

                if d2 <= 12.25 and _is_hbond_atom(pa) and _is_hbond_atom(la):
                    _add_pair_hit(by_type, "hbond", pa, la)
                    continue

                if d2 <= 14.44:
                    halogen_like = (la.elem in HALOGEN_ELEMS and _is_hbond_atom(pa)) or (
                        pa.elem in HALOGEN_ELEMS and _is_hbond_atom(la)
                    )
                    if halogen_like:
                        _add_pair_hit(by_type, "halogen", pa, la)
                        continue

                if d2 <= 17.64 and _is_hydrophobic_atom(pa) and _is_hydrophobic_atom(la):
                    _add_pair_hit(by_type, "hydrophobic", pa, la)

    aromatic_payload = by_type["aromatic"]
    aromatic_payload["subtypes"] = {"parallel": 0, "tshape": 0}

    for pr in protein_rings:
        for lr in ligand_rings:
            centroid_dist = float(np.linalg.norm(pr.centroid - lr.centroid))
            if centroid_dist > centroid_cutoff:
                continue

            angle = _angle_between_normals_deg(pr.normal, lr.normal)
            is_parallel = angle <= parallel_angle_max
            is_tshape = tshape_angle_min <= angle <= tshape_angle_max
            if not (is_parallel or is_tshape):
                continue

            pa, la = _nearest_pair(protein_atoms, ligand_atoms, pr.atom_serials, lr.atom_serials)
            _add_pair_hit(by_type, "aromatic", pa, la)
            aromatic_payload["proteinAtomSerials"].update(pr.atom_serials)
            aromatic_payload["ligandAtomSerials"].update(lr.atom_serials)
            aromatic_payload["residueKeys"].add(pr.residue_key)
            if is_parallel:
                aromatic_payload["subtypes"]["parallel"] += 1
            if is_tshape:
                aromatic_payload["subtypes"]["tshape"] += 1

    return by_type


def _serialize_payload(payload: Dict[str, object]) -> Dict[str, object]:
    out = dict(payload)
    out["proteinAtomSerials"] = sorted(out["proteinAtomSerials"])
    out["ligandAtomSerials"] = sorted(out["ligandAtomSerials"])
    out["residueKeys"] = sorted(out["residueKeys"])
    return out


def compute_sdf_interactions(
    protein_pdb: Path,
    ligand_sdf: Path,
    *,
    include_non_pi: bool,
    centroid_cutoff: float,
    parallel_angle_max: float,
    tshape_angle_min: float,
    tshape_angle_max: float,
) -> List[Dict[str, object]]:
    protein_atoms = parse_pdb_atoms(protein_pdb)
    protein_rings = build_protein_aromatic_rings(protein_atoms)

    supplier = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)
    results: List[Dict[str, object]] = []
    for idx, mol in enumerate(supplier):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        ligand_atoms, ligand_rings = ligand_atoms_and_rings(mol)
        interactions = classify_interactions(
            protein_atoms,
            ligand_atoms,
            protein_rings,
            ligand_rings,
            include_non_pi=include_non_pi,
            centroid_cutoff=centroid_cutoff,
            parallel_angle_max=parallel_angle_max,
            tshape_angle_min=tshape_angle_min,
            tshape_angle_max=tshape_angle_max,
        )

        pose_name = ""
        if mol.HasProp("_Name"):
            pose_name = mol.GetProp("_Name").strip()

        results.append(
            {
                "index": idx,
                "name": pose_name,
                "interactions": {k: _serialize_payload(v) for k, v in interactions.items()},
            }
        )
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute protein-ligand interactions with rigorous pi-pi geometry"
    )
    parser.add_argument("--protein-pdb", required=True, type=Path, help="Protein structure in PDB format")
    parser.add_argument("--ligand-sdf", required=True, type=Path, help="Ligand poses in SDF format")
    parser.add_argument("--output-json", required=True, type=Path, help="Output JSON path")
    parser.add_argument("--centroid-cutoff", type=float, default=6.0, help="Pi-pi centroid cutoff in A")
    parser.add_argument(
        "--parallel-angle-max",
        type=float,
        default=30.0,
        help="Max plane angle for parallel pi-pi interactions (degrees)",
    )
    parser.add_argument(
        "--tshape-angle-min",
        type=float,
        default=60.0,
        help="Min plane angle for T-shaped pi-pi interactions (degrees)",
    )
    parser.add_argument(
        "--tshape-angle-max",
        type=float,
        default=120.0,
        help="Max plane angle for T-shaped pi-pi interactions (degrees)",
    )
    parser.add_argument(
        "--pi-only",
        action="store_true",
        help="Only compute aromatic interactions; skip non-pi interaction classes",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    results = compute_sdf_interactions(
        args.protein_pdb,
        args.ligand_sdf,
        include_non_pi=not args.pi_only,
        centroid_cutoff=args.centroid_cutoff,
        parallel_angle_max=args.parallel_angle_max,
        tshape_angle_min=args.tshape_angle_min,
        tshape_angle_max=args.tshape_angle_max,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "protein_pdb": str(args.protein_pdb),
                "ligand_sdf": str(args.ligand_sdf),
                "molecule_count": len(results),
                "results": results,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
