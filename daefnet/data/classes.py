"""Class lists used by the data sources.

Synthetic labels are written as primitives joined by "+", for example
"sag+harmonics+oscillatory". The primitives are those of Table 4.
"""

PRIMITIVES = (
    "sag",
    "swell",
    "interruption",
    "impulsive",
    "oscillatory",
    "harmonics",
    "flicker",
    "notch",
)

# D2: 9 single classes, 14 two-component and 6 three-component combinations.
D2_CLASSES = (
    "normal",
    "sag",
    "swell",
    "interruption",
    "impulsive",
    "oscillatory",
    "harmonics",
    "flicker",
    "notch",
    "sag+harmonics",
    "swell+harmonics",
    "interruption+harmonics",
    "flicker+harmonics",
    "sag+oscillatory",
    "swell+oscillatory",
    "harmonics+oscillatory",
    "sag+impulsive",
    "swell+impulsive",
    "harmonics+impulsive",
    "flicker+oscillatory",
    "sag+flicker",
    "sag+notch",
    "harmonics+notch",
    "sag+harmonics+oscillatory",
    "swell+harmonics+oscillatory",
    "sag+harmonics+flicker",
    "swell+harmonics+flicker",
    "sag+harmonics+impulsive",
    "interruption+harmonics+oscillatory",
)

# D4: classes injected into the multi-bus circuit (Section 3.1).
D4_CLASSES = (
    "sag",
    "swell",
    "interruption",
    "oscillatory",
    "harmonics",
    "flicker",
    "sag+harmonics",
    "swell+harmonics",
    "flicker+harmonics",
)

# D1: class order of SEED-PQD-v1 as distributed on Zenodo (record 11843312).
D1_CLASSES = (
    "Pure_Sinusoidal",
    "Sag",
    "Swell",
    "Interruption",
    "Transient",
    "Oscillatory_Transient",
    "Harmonics",
    "Harmonics_with_Sag",
    "Harmonics_with_Swell",
    "Flicker",
    "Flicker_with_Sag",
    "Flicker_with_Swell",
    "Sag_with_Oscillatory_Transient",
    "Swell_with_Oscillatory_Transient",
    "Sag_with_Harmonics",
    "Swell_with_Harmonics",
    "Notch",
)

# D3: categories after mapping the GESL taxonomy (Section 3.1).
D3_CLASSES = (
    "fault_induced_sag",
    "motor_starting_sag",
    "swell",
    "momentary_interruption",
    "capacitor_switching_transient",
    "incipient_arcing_fault",
    "frequency_excursion",
    "oscillation",
)


def components(label):
    """Return the primitives that make up a synthetic class label."""
    if label == "normal":
        return ()
    parts = tuple(label.split("+"))
    for part in parts:
        if part not in PRIMITIVES:
            raise ValueError(f"unknown primitive '{part}' in label '{label}'")
    return parts


assert len(D2_CLASSES) == 29
assert len(D4_CLASSES) == 9
assert len(D1_CLASSES) == 17
