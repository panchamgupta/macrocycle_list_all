#!/usr/bin/env python3
import argparse

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


def _count_hbd_lipinski(mol):
    """Count H-bond donors using Lipinski definition: NH2 = 2, NH/OH = 1 each."""
    mol_h = Chem.AddHs(Chem.Mol(mol))
    hbd = 0
    for atom in mol_h.GetAtoms():
        if atom.GetSymbol() == 'N' and atom.GetTotalDegree() <= 3:
            explicit_h = sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H')
            if explicit_h >= 1:
                hbd += explicit_h
        elif atom.GetSymbol() == 'O' and atom.GetTotalDegree() <= 2:
            explicit_h = sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H')
            if explicit_h == 1:
                hbd += 1
    return hbd


def _compute_rdkit_props(mol):
    """Return a dict of RDKit descriptors for a molecule."""
    return {
        "MW": Descriptors.MolWt(mol),
        "ExactMW": Descriptors.ExactMolWt(mol),
        "cLogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": _count_hbd_lipinski(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotBonds": Lipinski.NumRotatableBonds(mol),
        "HeavyAtoms": Descriptors.HeavyAtomCount(mol),
        "FormalCharge": Chem.GetFormalCharge(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
    }


def add_csv_props_to_sdf(input_sdf, input_csv, output_sdf, sdf_id_field="_Name", csv_id_field="ID"):
    # Read CSV as strings to preserve formatting exactly as provided.
    df = pd.read_csv(input_csv, dtype=str)

    if csv_id_field not in df.columns:
        raise ValueError(
            f"CSV ID column '{csv_id_field}' not found. Available columns: {list(df.columns)}"
        )

    # Build lookup by molecule ID. If IDs repeat, keep the first row per ID.
    dup_mask = df.duplicated(subset=[csv_id_field], keep="first")
    duplicate_count = int(dup_mask.sum())
    if duplicate_count:
        print(
            f"Warning: found {duplicate_count} duplicate rows for '{csv_id_field}'. "
            "Using the first occurrence for each ID."
        )
    df_unique = df.drop_duplicates(subset=[csv_id_field], keep="first")
    csv_dict = df_unique.set_index(csv_id_field).to_dict(orient="index")

    suppl = Chem.SDMolSupplier(input_sdf, removeHs=False)
    writer = Chem.SDWriter(output_sdf)

    matched = 0
    total = 0
    failed = 0
    rdkit_prop_failures = 0

    for mol in suppl:
        if mol is None:
            failed += 1
            continue

        total += 1

        if sdf_id_field == "_Name":
            mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        else:
            mol_id = mol.GetProp(sdf_id_field) if mol.HasProp(sdf_id_field) else ""

        if mol_id in csv_dict:
            matched += 1
            for prop_name, prop_value in csv_dict[mol_id].items():
                if pd.notna(prop_value):
                    mol.SetProp(str(prop_name), str(prop_value))

        # Compute and attach RDKit descriptor tags for every valid molecule.
        try:
            rdkit_props = _compute_rdkit_props(mol)
            for prop_name, prop_value in rdkit_props.items():
                if isinstance(prop_value, float):
                    mol.SetProp(prop_name, f"{prop_value:.6f}")
                else:
                    mol.SetProp(prop_name, str(prop_value))
        except Exception:
            rdkit_prop_failures += 1

        writer.write(mol)

    writer.close()

    print("Done.")
    print(f"Total molecules processed: {total}")
    print(f"Matched molecules: {matched}")
    print(f"Unmatched molecules: {total - matched}")
    if failed:
        print(f"Failed molecules skipped: {failed}")
    if rdkit_prop_failures:
        print(f"RDKit property failures: {rdkit_prop_failures}")
    print(f"Output written to: {output_sdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add CSV properties and computed RDKit descriptors as SDF tags using a shared molecule ID."
    )
    parser.add_argument("-sdf", required=True, help="Input SDF file")
    parser.add_argument("-csv", required=True, help="Input CSV file")
    parser.add_argument("-out", required=True, help="Output SDF file")
    parser.add_argument(
        "--sdf_id_field",
        default="_Name",
        help="SDF ID field. Use _Name for title line.",
    )
    parser.add_argument("--csv_id_field", default="ID", help="CSV ID column name")

    args = parser.parse_args()

    add_csv_props_to_sdf(
        input_sdf=args.sdf,
        input_csv=args.csv,
        output_sdf=args.out,
        sdf_id_field=args.sdf_id_field,
        csv_id_field=args.csv_id_field,
    )
