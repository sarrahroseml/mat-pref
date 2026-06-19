"""Build a chemical descriptor lookup table for all element/site/structure combos.

Produces data/chem_descriptors.json with Shannon ionic radii, Pauling
electronegativity, d-electron count, and coordination preferences for
every (element, structure_family, site_label) triple in the dataset.

Usage:
    python scripts/build_chem_descriptors.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pymatgen.core import Element, Species

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Oxidation state & coordination number mapping
# ---------------------------------------------------------------------------
# (structure_family, site_label) -> (typical_oxidation_state, coordination_number)

SITE_OX_CN = {
    # Perovskite ABO3
    ("perovskite", "A-site"): (2, 12),
    ("perovskite", "B-site"): (4, 6),
    # Double perovskite A2BB'O6
    ("double_perovskite", "A-site"): (2, 12),
    ("double_perovskite", "B-site"): (3, 6),
    ("double_perovskite", "B'-site"): (5, 6),
    # Halide perovskite ABX3
    ("halide_perovskite", "A-site"): (1, 12),
    ("halide_perovskite", "B-site"): (2, 6),
    # Spinel AB2O4
    ("spinel", "A-site"): (2, 4),
    ("spinel", "B-site"): (3, 6),
    # Pyrochlore A2B2O7
    ("pyrochlore", "A-site"): (3, 8),
    ("pyrochlore", "B-site"): (4, 6),
    # Garnet A3B2(XO4)3
    ("garnet", "A-site"): (3, 8),
    ("garnet", "B-site"): (3, 6),
    # Olivine M2XO4
    ("olivine", "M-site"): (2, 6),
    ("olivine", "T-site"): (5, 4),
    # Scheelite ABO4
    ("scheelite", "A-site"): (2, 8),
    ("scheelite", "B-site"): (6, 4),
    # Brownmillerite A2B2O5
    ("brownmillerite", "A-site"): (2, 8),
    ("brownmillerite", "B-site"): (3, 6),  # mixed 6/4, use 6 as primary
    # Layered oxide ABO2
    ("layered_oxide", "A-site"): (1, 6),
    ("layered_oxide", "B-site"): (3, 6),
    # NASICON AxM2(XO4)3
    ("NASICON", "A-site"): (1, 6),
    ("NASICON", "B-site"): (3, 6),
}


def get_d_electron_count(element: str, oxidation_state: int) -> int:
    """Calculate d-electron count for an element at a given oxidation state."""
    el = Element(element)
    atomic_num = el.Z

    # Electron configuration: total electrons minus oxidation state
    remaining = atomic_num - oxidation_state

    if remaining <= 0:
        return 0

    # For transition metals (Z=21-30, 39-48, 57-80, 89+), count d-electrons
    # Simple approach: get the element's group and period to determine d-count
    # d-electrons = remaining electrons after filling up to the relevant d-shell

    # Noble gas cores
    cores = [
        (2, 0),    # He
        (10, 0),   # Ne
        (18, 0),   # Ar
        (36, 0),   # Kr
        (54, 0),   # Xe
        (86, 0),   # Rn
    ]

    # For transition metals, after removing electrons (highest n first),
    # the d-electron count is what remains in the d-shell
    # Simplified: use pymatgen's electronic structure

    # Common d-electron counts for transition metals
    # d-block: groups 3-12, periods 4-7
    if not el.is_transition_metal and element not in ('La', 'Ce', 'Pr', 'Nd',
        'Sm', 'Eu', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
        'Th', 'Pa', 'U', 'Np', 'Pu'):
        return 0

    # For d-block: d-electrons = (group number - oxidation state) for simple cases
    # More robust: count from electron configuration
    group = el.group
    if el.is_transition_metal:
        # 3d: period 4, groups 3-12 -> max 10 d-electrons
        # d^n where n = (Z - core) - 2 (for 4s2) adjusted for oxidation
        if 21 <= atomic_num <= 30:  # 3d
            d_count = max(0, atomic_num - 18 - 2 - oxidation_state + 2)
            return max(0, min(10, d_count))
        elif 39 <= atomic_num <= 48:  # 4d
            d_count = max(0, atomic_num - 36 - 2 - oxidation_state + 2)
            return max(0, min(10, d_count))
        elif 71 <= atomic_num <= 80:  # 5d (after lanthanides)
            d_count = max(0, atomic_num - 54 - 14 - 2 - oxidation_state + 2)
            return max(0, min(10, d_count))
    # Lanthanides: f-electrons, d-count effectively 0 for most
    if 57 <= atomic_num <= 71:
        # La-Lu: typically d^0 or d^1 in their common oxidation states
        if oxidation_state >= 3:
            return 0
        return max(0, 1 - (oxidation_state - 2))
    # Actinides
    if 89 <= atomic_num <= 103:
        return 0

    return 0


def get_coordination_preferences(element: str, oxidation_state: int) -> str:
    """Return common coordination preferences as a brief string."""
    el = Element(element)

    # Alkali metals
    if el.group == 1 and element != 'H':
        return "flexible; prefers high CN (6-12)"
    # Alkaline earth
    if el.group == 2:
        return "prefers CN 6-8; rigid size requirements"
    # Transition metals 3d
    if el.is_transition_metal:
        d_count = get_d_electron_count(element, oxidation_state)
        prefs = []
        if d_count in (0, 5, 10):
            prefs.append("no CFSE preference")
        elif d_count in (3, 8):
            prefs.append("strong octahedral preference")
        elif d_count in (4, 9):
            prefs.append("Jahn-Teller active")
        else:
            prefs.append("moderate octahedral preference")
        if d_count == 10:
            prefs.append("tetrahedral compatible")
        return "; ".join(prefs)
    # Lanthanides
    if 57 <= el.Z <= 71:
        return "prefers CN 8-12; lanthanide contraction"
    # Post-transition metals
    if el.group in (13, 14, 15):
        return "flexible CN; lone pair effects possible"
    # Halogens
    if el.group == 17:
        return "typically anionic (CN 1-6)"
    # Actinides
    if 89 <= el.Z <= 103:
        return "prefers high CN (8-12); multiple oxidation states"

    return "varies"


def _best_oxidation_state(element: str, target_ox: int) -> int:
    """Find the best oxidation state that has ionic radius data.

    Tries the target first, then common oxidation states for the element,
    preferring the one closest to the target.
    """
    el = Element(element)

    # Try target first
    try:
        sp = Species(element, target_ox)
        if sp.ionic_radius:
            return target_ox
    except Exception:
        pass

    # Fall back to common oxidation states, sorted by closeness to target
    common_ox = sorted(el.common_oxidation_states, key=lambda x: abs(x - target_ox))
    for ox in common_ox:
        if ox <= 0:
            continue
        try:
            sp = Species(element, ox)
            if sp.ionic_radius:
                return ox
        except Exception:
            continue

    # Last resort: return target anyway
    return target_ox


def build_descriptor(element: str, struct_family: str, site_label: str) -> dict:
    """Build a chemical descriptor dict for one element at a given site."""
    key = (struct_family, site_label)
    target_ox, cn = SITE_OX_CN.get(key, (None, None))

    if target_ox is None:
        return {"element": element, "error": f"Unknown site: {key}"}

    ox_state = _best_oxidation_state(element, target_ox)
    el = Element(element)
    result = {
        "element": element,
        "oxidation_state": ox_state,
        "coordination_number": cn,
        "pauling_electronegativity": el.X,
    }

    # Shannon ionic radius
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp = Species(element, ox_state)
            result["ionic_radius_A"] = float(sp.ionic_radius) if sp.ionic_radius else None
    except Exception:
        result["ionic_radius_A"] = None

    result["d_electron_count"] = get_d_electron_count(element, ox_state)
    result["coordination_preferences"] = get_coordination_preferences(element, ox_state)

    return result


def format_descriptor_text(desc: dict) -> str:
    """Format a descriptor dict as a concise text string for prompting."""
    parts = [f"{desc['element']}"]
    ox = desc.get("oxidation_state")
    if ox is not None:
        parts[0] += f" ({'+' if ox > 0 else ''}{ox})"

    r = desc.get("ionic_radius_A")
    if r is not None:
        parts.append(f"ionic radius={r:.3f} Å")
    else:
        parts.append("ionic radius=N/A")

    en = desc.get("pauling_electronegativity")
    if en is not None:
        parts.append(f"electronegativity={en:.2f}")

    d = desc.get("d_electron_count", 0)
    parts.append(f"d-electrons={d}")

    cn = desc.get("coordination_number")
    if cn is not None:
        parts.append(f"CN={cn}")

    pref = desc.get("coordination_preferences", "")
    if pref:
        parts.append(f"coord. pref: {pref}")

    return "; ".join(parts)


def main():
    # Load all questions
    all_qs = []
    for fname in ["train.jsonl", "test_iid.jsonl", "test_ood_host.jsonl",
                   "test_ood_property.jsonl"]:
        path = DATA_DIR / "questions" / fname
        if path.exists():
            with open(path) as f:
                all_qs.extend([json.loads(line) for line in f if line.strip()])

    # Collect unique (element, structure_family, site_label) triples
    triples = set()
    for q in all_qs:
        for c in q["candidates"]:
            triples.add((c["element"], q["structure_family"], q["site_label"]))

    print(f"Building descriptors for {len(triples)} unique (element, struct, site) triples...")

    # Build lookup: keyed by "element|structure_family|site_label"
    lookup = {}
    missing_radius = 0
    for element, struct, site in sorted(triples):
        desc = build_descriptor(element, struct, site)
        key = f"{element}|{struct}|{site}"
        lookup[key] = desc
        if desc.get("ionic_radius_A") is None:
            missing_radius += 1

    print(f"Built {len(lookup)} descriptors ({missing_radius} missing ionic radius)")

    # Save
    out_path = DATA_DIR / "chem_descriptors.json"
    with open(out_path, "w") as f:
        json.dump(lookup, f, indent=2)
    print(f"Saved to {out_path}")

    # Print a few examples
    print("\nExamples:")
    for key in list(lookup.keys())[:5]:
        desc = lookup[key]
        print(f"  {key}: {format_descriptor_text(desc)}")


if __name__ == "__main__":
    main()
