"""
fix_provenance_romania.py
Fixes the provenance.yaml where Romania section got embedded inside
Nakhchivan pTransferLimit notes. Moves Romania to proper top-level position.
"""
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "epm/input/data_blacksea/provenance.yaml"
txt = p.read_text(encoding="utf-8")

# The file has:
# ...
# pTransferLimit (Nakhchivan):
#   notes: >
#     Three connections for Nakhchivan:
#
# Romania:          <--- should NOT be here
#   ...Romania provenance...
#   last_updated: 2026-06-11
#           (1) Nakhchivan ...  <--- these are the real Nakhchivan notes
#   confidence: medium
#   ...
#   last_updated: 2026-06-10
#
# We need to:
# 1. Move Romania: section after the Nakhchivan section closes
# 2. Put the (1)(2)(3) Nakhchivan notes where they belong (after "Three connections...")

MARKER_NAKHCHIVAN_NOTES_START = "          Three connections for Nakhchivan:\n"
MARKER_ROMANIA_START = "\nRomania:\n"
MARKER_AFTER_ROMANIA = "\n    last_updated: 2026-06-11\n          (1) Nakhchivan"
MARKER_NAKHCHIVAN_END = "\n    last_updated: 2026-06-10\n"

# Find positions
idx_notes_start = txt.find(MARKER_NAKHCHIVAN_NOTES_START)
idx_blank_before_ro = txt.find("\n\nRomania:\n", idx_notes_start)  # blank + Romania:
idx_after_ro = txt.find(MARKER_AFTER_ROMANIA, idx_blank_before_ro)
idx_nakhchivan_close = txt.find(MARKER_NAKHCHIVAN_END, idx_after_ro)

assert idx_notes_start != -1, "Could not find Nakhchivan notes start"
assert idx_blank_before_ro != -1, "Could not find misplaced Romania:"
assert idx_after_ro != -1, "Could not find end of Romania section"
assert idx_nakhchivan_close != -1, "Could not find Nakhchivan section end"

# Extract the Romania section (from "\n\nRomania:\n" to just before "(1) Nakhchivan")
# idx_blank_before_ro points to the first \n of "\n\nRomania:\n"
# idx_after_ro + len("\n    last_updated: 2026-06-11\n") is where "(1) Nakhchivan" starts
end_of_last_updated = idx_after_ro + len("\n    last_updated: 2026-06-11\n")
romania_section = txt[idx_blank_before_ro:end_of_last_updated]

# The Nakhchivan (1)(2)(3) content: from "(1) Nakhchivan" to end of pTransferLimit
nakhchivan_notes_tail = txt[end_of_last_updated : idx_nakhchivan_close + len(MARKER_NAKHCHIVAN_END)]

# Build corrected text:
# Part 1: everything up to (and including) "Three connections for Nakhchivan:\n"
part1 = txt[:idx_notes_start + len(MARKER_NAKHCHIVAN_NOTES_START)]

# Part 2: the Nakhchivan (1)(2)(3) notes tail (= the real notes content)
part2 = nakhchivan_notes_tail

# Part 3: the Romania section (moved to after pTransferLimit closes)
part3 = romania_section

# Assemble
new_txt = part1 + part2 + part3

p.write_text(new_txt, encoding="utf-8")
print("Fixed provenance.yaml — Romania section moved to correct position.")

# Verify structure
lines = new_txt.splitlines()
for i, line in enumerate(lines):
    if "Romania:" in line and not line.strip().startswith("#"):
        print(f"  Line {i+1}: {line!r}")
print(f"Total lines: {len(lines)}")
