# WaterKit demo: receptor + sampled explicit waters
# Open with:  pymol demo/visualize.pml   (run from the repo root)

reinitialize
bg_color white
set ray_opaque_background, 0

# --- load ---
load demo/receptor.pdb, receptor
load demo/waters.pdb, waters

hide everything

# --- receptor as sticks ---
show sticks, receptor
set stick_radius, 0.14, receptor
color grey60, receptor and elem C
util.cnc receptor
set valence, 0

# --- sampled waters: red O spheres + thin O-H sticks ---
show spheres, waters and elem O
set sphere_scale, 0.30, waters and elem O
show sticks, waters
set stick_radius, 0.06, waters
color red, waters and elem O
color white, waters and elem H

# --- predicted hydrogen bonds: receptor polar atoms <-> water O (<= 3.5 A) ---
distance hbonds, (receptor and elem N+O), (waters and elem O), 3.5
color marine, hbonds
hide labels, hbonds
set dash_gap, 0.35
set dash_width, 2.5

# --- water-water contacts (first shell network), lighter ---
distance wwnet, (waters and elem O), (waters and elem O), 3.4
color grey80, wwnet
hide labels, wwnet

# --- view ---
set sphere_quality, 3
set stick_quality, 12
orient
zoom all, 3
turn y, 20
turn x, -10
