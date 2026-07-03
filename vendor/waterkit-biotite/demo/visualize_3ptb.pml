# WaterKit on trypsin-benzamidine (3PTB): predicted vs crystallographic waters
# Open with:  pymol demo/visualize_3ptb.pml   (run from the repo root)

reinitialize
bg_color white
set ray_opaque_background, 0

load demo/3ptb.pdb, xtal
load demo/3ptb_waters.pdb, wk

hide everything

# --- protein: translucent cartoon ---
show cartoon, xtal and polymer
color grey80, xtal and polymer
set cartoon_transparency, 0.45

# --- benzamidine ligand: yellow sticks ---
show sticks, xtal and resn BEN
color yellow, (xtal and resn BEN and elem C)
util.cnc (xtal and resn BEN)
set stick_radius, 0.18, xtal and resn BEN

# --- binding-site residues near the ligand: thin lines ---
select site, byres (xtal and polymer within 4.5 of (xtal and resn BEN))
show lines, site
color cyan, (site and elem C)
util.cnc site

# --- crystallographic waters (experimental): orange spheres ---
show spheres, xtal and resn HOH
set sphere_scale, 0.22, xtal and resn HOH
color orange, xtal and resn HOH

# --- WaterKit predicted waters: red O spheres + white H ---
show spheres, wk and elem O
set sphere_scale, 0.28, wk and elem O
show sticks, wk
set stick_radius, 0.05, wk
color red, wk and elem O
color white, wk and elem H

# --- predicted-water H-bonds to ligand + site (<=3.5 A) ---
distance hbonds, (wk and elem O), ((xtal and resn BEN) or (site and (elem N+O))), 3.5
color marine, hbonds
hide labels, hbonds
set dash_gap, 0.35
set dash_width, 2.5

# --- focus on the S1 pocket / ligand ---
set sphere_quality, 3
orient (xtal and resn BEN)
zoom (xtal and resn BEN), 9
