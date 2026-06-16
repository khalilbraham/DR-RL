"""Synthetic compartment library with ground-truth equivalence/identifiability."""

from drrl.data.synth.library import (
    ModelPair,
    SyntheticCase,
    confounded,
    generate_cases,
    indistinguishable_pairs,
    michaelis_menten,
    one_compartment,
    tmdd,
    two_compartment,
    two_compartment_macro,
)

__all__ = [
    "ModelPair",
    "SyntheticCase",
    "confounded",
    "generate_cases",
    "indistinguishable_pairs",
    "michaelis_menten",
    "one_compartment",
    "tmdd",
    "two_compartment",
    "two_compartment_macro",
]
