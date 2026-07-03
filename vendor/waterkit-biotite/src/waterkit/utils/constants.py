"""Physical constants and package-wide defaults."""

# Boltzmann constant in kcal/mol/K (matches the value used by waterkit/AutoDock).
KB = 0.0019872041

# Default sampling temperature in Kelvin.
DEFAULT_TEMPERATURE = 300.0

# Default AutoGrid spacing in Angstrom.
DEFAULT_SPACING = 0.375

# Geometric constraints on water placement around a hydrogen-bond anchor (Angstrom / degrees).
MIN_HB_DISTANCE = 2.5
MAX_HB_DISTANCE = 3.6
HB_ANGLE = 90.0

# Supported water models.
WATER_MODELS = ("tip3p", "tip5p")

# Input structure formats handled by the loader.
SUPPORTED_FORMATS = (".pdb", ".pdbqt", ".cif", ".bcif", ".mmcif")
