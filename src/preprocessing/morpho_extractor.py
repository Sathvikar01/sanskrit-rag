"""Morphological feature extraction from annotated Sanskrit text."""

import re
from collections import Counter
from dataclasses import dataclass, field

from src.utils.logger import logger


@dataclass
class MorphologicalProfile:
    """Aggregated morphological features for a text chunk."""

    # Distribution of grammatical features
    case_distribution: dict[str, int] = field(default_factory=dict)
    gender_distribution: dict[str, int] = field(default_factory=dict)
    number_distribution: dict[str, int] = field(default_factory=dict)
    tense_distribution: dict[str, int] = field(default_factory=dict)
    mood_distribution: dict[str, int] = field(default_factory=dict)
    verbform_distribution: dict[str, int] = field(default_factory=dict)

    # Raw lemmas extracted
    lemmas: list[str] = field(default_factory=list)
    roots: list[str] = field(default_factory=list)

    # Counts
    total_tokens: int = 0
    noun_count: int = 0
    verb_count: int = 0


def parse_morpho_token(token: str) -> dict[str, str]:
    """Parse a morphological annotation token.

    Format: lemma_Feature1=Val1|Feature2=Val2|...
    Example: dharma_Case=Loc|Gender=Neut|Number=Sing

    Returns:
        Dictionary with 'lemma' and feature key-value pairs.
    """
    result = {"lemma": "", "features": {}}

    if not token or not token.strip():
        return result

    token = token.strip()

    # Handle tokens without features (e.g., "ca_", "eva_")
    if "_" in token:
        parts = token.split("_", 1)
        lemma = parts[0]
        feature_str = parts[1] if len(parts) > 1 else ""
    else:
        lemma = token
        feature_str = ""

    result["lemma"] = lemma

    if feature_str:
        # Parse features like "Case=Loc|Gender=Neut|Number=Sing"
        for feature in feature_str.split("|"):
            feature = feature.strip()
            if "=" in feature:
                key, value = feature.split("=", 1)
                result["features"][key.strip()] = value.strip()

    return result


def extract_morpho_features_from_line(line: str) -> list[dict[str, str]]:
    """Extract morphological features from a single annotated line.

    Args:
        line: Space-separated morphological tokens.

    Returns:
        List of parsed token dictionaries.
    """
    tokens = line.split()
    return [parse_morpho_token(t) for t in tokens if t.strip()]


def build_morphological_profile(annotated_lines: list[str]) -> MorphologicalProfile:
    """Build an aggregated morphological profile from annotated lines.

    Args:
        annotated_lines: List of morphologically annotated text lines.

    Returns:
        MorphologicalProfile with aggregated feature distributions.
    """
    profile = MorphologicalProfile()

    case_counter = Counter()
    gender_counter = Counter()
    number_counter = Counter()
    tense_counter = Counter()
    mood_counter = Counter()
    verbform_counter = Counter()

    for line in annotated_lines:
        tokens = extract_morpho_features_from_line(line)

        for token_data in tokens:
            lemma = token_data["lemma"]
            features = token_data["features"]

            if lemma:
                profile.lemmas.append(lemma)

            if not features:
                continue

            profile.total_tokens += 1

            if "Case" in features:
                case_counter[features["Case"]] += 1
            if "Gender" in features:
                gender_counter[features["Gender"]] += 1
            if "Number" in features:
                number_counter[features["Number"]] += 1
            if "Tense" in features:
                tense_counter[features["Tense"]] += 1
            if "Mood" in features:
                mood_counter[features["Mood"]] += 1
            if "VerbForm" in features:
                verbform_counter[features["VerbForm"]] += 1

            is_noun = "Case" in features and "Gender" in features
            is_verb = "Tense" in features or "Mood" in features or "VerbForm" in features

            if is_noun:
                profile.noun_count += 1
            if is_verb:
                profile.verb_count += 1

    profile.case_distribution = dict(case_counter)
    profile.gender_distribution = dict(gender_counter)
    profile.number_distribution = dict(number_counter)
    profile.tense_distribution = dict(tense_counter)
    profile.mood_distribution = dict(mood_counter)
    profile.verbform_distribution = dict(verbform_counter)

    profile.lemmas = list(set(profile.lemmas))

    return profile


def extract_lemmas_from_segmentation_line(line: str) -> list[dict[str, str]]:
    """Extract lemma pairs from a segmentation-lemma line.

    Format: surface_lemma surface_lemma ...
    Example: dhṛtarāṣṭraḥ_dhṛtarāṣṭra uvāca_vac

    Returns:
        List of dicts with 'surface' and 'lemma' keys.
    """
    results = []
    tokens = line.split()

    for token in tokens:
        if not token.strip():
            continue

        if "_" in token:
            parts = token.rsplit("_", 1)
            surface = parts[0].rstrip("-")  # Remove compound boundary marker
            lemma = parts[1]
            results.append({"surface": surface, "lemma": lemma})

    return results


def extract_all_lemmas_from_segmentation(
    segmentation_lines: list[str],
) -> list[dict[str, str]]:
    """Extract all lemma pairs from multiple segmentation lines."""
    all_pairs = []
    for line in segmentation_lines:
        all_pairs.extend(extract_lemmas_from_segmentation_line(line))
    return all_pairs


def compute_morpho_similarity(
    profile1: MorphologicalProfile,
    profile2: MorphologicalProfile,
) -> float:
    """Compute similarity between two morphological profiles.

    Uses cosine similarity on feature distribution vectors.

    Returns:
        Similarity score between 0 and 1.
    """
    if profile1.total_tokens == 0 or profile2.total_tokens == 0:
        return 0.0

    all_cases = set(profile1.case_distribution.keys()) | set(
        profile2.case_distribution.keys()
    )
    all_genders = set(profile1.gender_distribution.keys()) | set(
        profile2.gender_distribution.keys()
    )
    all_numbers = set(profile1.number_distribution.keys()) | set(
        profile2.number_distribution.keys()
    )

    def cosine_sim(vec1: list[float], vec2: list[float]) -> float:
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    vec1 = []
    vec2 = []

    for case in sorted(all_cases):
        vec1.append(profile1.case_distribution.get(case, 0))
        vec2.append(profile2.case_distribution.get(case, 0))

    for gender in sorted(all_genders):
        vec1.append(profile1.gender_distribution.get(gender, 0))
        vec2.append(profile2.gender_distribution.get(gender, 0))

    for number in sorted(all_numbers):
        vec1.append(profile1.number_distribution.get(number, 0))
        vec2.append(profile2.number_distribution.get(number, 0))

    return cosine_sim(vec1, vec2)
