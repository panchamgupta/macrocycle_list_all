#!/usr/bin/env bash

if command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n rdkit-env python)
else
  PYTHON_CMD=(python3)
fi

"${PYTHON_CMD[@]}" process_docking_IF_show_docking.py --input  5C_2210_docking_poses.sdf \
	--protein-pdb 2210_xtal.mol2 \
	--file-prefix B3G_5C_beta_aa_042826 \
	--interaction-csv IF_5C_after_docking_filtered.csv \
	--interaction-id-col Title \
  	--interaction-count-col interaction_count \
	--outdir /home/pgupta11/Projects/B3GNT2/macrocycles/docking_panel/ \
	--score-props r_i_docking_score fsp3 interaction_count druglike_score \
	--ref-ligand-sdf 2210_xtal_lig.sdf \
	--n-workers 8 \
	#--generate-all-mol-images \

