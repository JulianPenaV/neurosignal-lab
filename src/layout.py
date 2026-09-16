"""
Schematic 2D scalp layout for the 64-channel BCI2000/EEGMMIDB montage, derived
purely from the standard 10-10 electrode naming convention (row = anterior-
posterior ring, position within row = left-to-right order). This is NOT a true
spherical/azimuthal projection of real 3D electrode coordinates -- it's a
topologically-correct schematic (rows in the right anterior-posterior order,
left/right/midline in the right order within each row) used only to make CSP
spatial patterns visually interpretable. Distances between rows/columns are not
meaningful; left-right symmetry and front-back ordering are.
"""

# Anterior (top) to posterior (bottom) rings, each already left-to-right.
ROWS = [
    ["Fp1", "Fpz", "Fp2"],
    ["Af7", "Af3", "Afz", "Af4", "Af8"],
    ["F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8"],
    ["Ft7", "Fc5", "Fc3", "Fc1", "Fcz", "Fc2", "Fc4", "Fc6", "Ft8"],
    ["T9", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "T10"],
    ["Tp7", "Cp5", "Cp3", "Cp1", "Cpz", "Cp2", "Cp4", "Cp6", "Tp8"],
    ["P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8"],
    ["Po7", "Po3", "Poz", "Po4", "Po8"],
    ["O1", "Oz", "O2"],
    ["Iz"],
]


def build_layout(channel_names):
    """
    channel_names: iterable of channel labels (any case) present in the data.
    Returns {channel_name: (x, y)} for every name found in ROWS, using each
    channel's *original* casing from channel_names.
    """
    name_by_lower = {c.lower(): c for c in channel_names}
    positions = {}
    for row_idx, row in enumerate(ROWS):
        n = len(row)
        for col_idx, ch in enumerate(row):
            key = ch.lower()
            if key not in name_by_lower:
                continue
            x = col_idx - (n - 1) / 2.0
            y = -row_idx
            positions[name_by_lower[key]] = (x, y)
    return positions
