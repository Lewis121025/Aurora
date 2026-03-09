from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class SeedPlotSpec:
    seed_id: str
    text: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeedPlotSpec":
        return cls(
            seed_id=str(data["seed_id"]),
            text=str(data["text"]),
        )


@dataclass(frozen=True)
class TraitPrior:
    name: str
    description: str
    alpha: float
    beta: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraitPrior":
        return cls(
            name=str(data["name"]),
            description=str(data["description"]),
            alpha=float(data["alpha"]),
            beta=float(data["beta"]),
        )


@dataclass(frozen=True)
class IntuitionAnchor:
    anchor_id: str
    text: str
    keywords: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntuitionAnchor":
        return cls(
            anchor_id=str(data["anchor_id"]),
            text=str(data["text"]),
            keywords=[str(item) for item in data.get("keywords", [])],
        )


@dataclass(frozen=True)
class PersonalityProfile:
    profile_id: str
    display_name: str
    identity_statement: str
    identity_narrative: str
    seed_narrative: str
    core_values: List[str]
    trait_priors: List[TraitPrior]
    seed_plots: List[SeedPlotSpec]
    intuition_anchors: List[IntuitionAnchor]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonalityProfile":
        return cls(
            profile_id=str(data["profile_id"]),
            display_name=str(data["display_name"]),
            identity_statement=str(data["identity_statement"]),
            identity_narrative=str(data["identity_narrative"]),
            seed_narrative=str(data["seed_narrative"]),
            core_values=[str(value) for value in data.get("core_values", [])],
            trait_priors=[TraitPrior.from_dict(item) for item in data.get("trait_priors", [])],
            seed_plots=[SeedPlotSpec.from_dict(item) for item in data.get("seed_plots", [])],
            intuition_anchors=[IntuitionAnchor.from_dict(item) for item in data.get("intuition_anchors", [])],
        )


def load_personality_profile(profile_id: str) -> PersonalityProfile:
    if profile_id != "aurora-v2-native":
        raise ValueError(f"unknown personality profile: {profile_id}")

    profile_path = Path(__file__).with_name("default_profile.json")
    with profile_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return PersonalityProfile.from_dict(payload)
