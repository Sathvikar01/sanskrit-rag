"""Concept extractor for Bhagavad Gita philosophical themes."""

import re
from dataclasses import dataclass, field

from src.utils.logger import logger


@dataclass
class Concept:
    """A philosophical concept from the Bhagavad Gita."""

    name_iast: str
    name_devanagari: str
    name_english: str
    description: str
    aliases_iast: list[str] = field(default_factory=list)
    aliases_devanagari: list[str] = field(default_factory=list)
    related_concepts: list[str] = field(default_factory=list)


# Seed concept dictionary for Bhagavad Gita
SEED_CONCEPTS: list[Concept] = [
    Concept(
        name_iast="dharma",
        name_devanagari="धर्म",
        name_english="duty/righteousness",
        description="Fundamental concept of cosmic order, duty, and righteous conduct",
        aliases_iast=["svadharma", "sadharma", "dharmya"],
        related_concepts=["karma", "yajna", "niyama"],
    ),
    Concept(
        name_iast="karma",
        name_devanagari="कर्म",
        name_english="action",
        description="Action or deed; the law of cause and effect",
        aliases_iast=["karmaphala", "karmayoga", "nishkamakarma", "akarma"],
        related_concepts=["dharma", "yajna", "moksha"],
    ),
    Concept(
        name_iast="bhakti",
        name_devanagari="भक्ति",
        name_english="devotion",
        description="Loving devotion to the Supreme; one of the three main yogas",
        aliases_iast=["bhaktiyoga", "bhagavadbhakti"],
        related_concepts=["prapatti", "ishvara", "moksha"],
    ),
    Concept(
        name_iast="jnana",
        name_devanagari="ज्ञान",
        name_english="knowledge",
        description="Spiritual knowledge or wisdom; discriminative knowledge of Self",
        aliases_iast=["jnanayoga", "atmajnana", "brahmajnana", "vijnana"],
        related_concepts=["buddhi", "atman", "brahman"],
    ),
    Concept(
        name_iast="yoga",
        name_devanagari="योग",
        name_english="union/discipline",
        description="Union with the divine; spiritual discipline and practice",
        aliases_iast=["yogayoga", "sannyasa", "tapas"],
        related_concepts=["dhyana", "samadhi", "karma"],
    ),
    Concept(
        name_iast="atman",
        name_devanagari="आत्मन्",
        name_english="self/soul",
        description="The eternal, unchanging Self; the individual soul",
        aliases_iast=["atma", "jivatman", "jiva", "purusha"],
        related_concepts=["brahman", "sharira", "moksha"],
    ),
    Concept(
        name_iast="brahman",
        name_devanagari="ब्रह्मन्",
        name_english="absolute",
        description="The ultimate reality; the Supreme Absolute",
        aliases_iast=["brahma", "parabrahman", "paramatman"],
        related_concepts=["atman", "ishvara", "moksha"],
    ),
    Concept(
        name_iast="ishvara",
        name_devanagari="ईश्वर",
        name_english="lord/god",
        description="The Supreme Lord; the personal aspect of God",
        aliases_iast=["bhagavan", "parameshvara", "deva"],
        related_concepts=["brahman", "bhakti", "prakriti"],
    ),
    Concept(
        name_iast="prakriti",
        name_devanagari="प्रकृति",
        name_english="nature/matter",
        description="Material nature; the three gunas; the manifest world",
        aliases_iast=["maya", "gunas", "pradhana"],
        related_concepts=["purusha", "gunas", "samsara"],
    ),
    Concept(
        name_iast="moksha",
        name_devanagari="मोक्ष",
        name_english="liberation",
        description="Liberation from the cycle of birth and death; final emancipation",
        aliases_iast=["mukti", "kaivalya", "apavarga", "nirvana"],
        related_concepts=["atman", "brahman", "samsara"],
    ),
    Concept(
        name_iast="samsara",
        name_devanagari="संसार",
        name_english="cycle of existence",
        description="The cycle of birth, death, and rebirth",
        aliases_iast=["bhavasagara", "janmamrityu"],
        related_concepts=["karma", "moksha", "maya"],
    ),
    Concept(
        name_iast="gunas",
        name_devanagari="गुण",
        name_english="qualities",
        description="The three qualities of material nature: sattva, rajas, tamas",
        aliases_iast=["sattva", "rajas", "tamas", "triguna", "gunatraya"],
        related_concepts=["prakriti", "karma", "yoga"],
    ),
    Concept(
        name_iast="kshetra",
        name_devanagari="क्षेत्र",
        name_english="field",
        description="The body as the field of action; the material field",
        aliases_iast=["kshetrajna", "sharira"],
        related_concepts=["atman", "prakriti", "purusha"],
    ),
    Concept(
        name_iast="ahimsa",
        name_devanagari="अहिंसा",
        name_english="non-violence",
        description="Non-violence in thought, word, and deed",
        aliases_iast=["avihimsa"],
        related_concepts=["dharma", "yama", "satya"],
    ),
    Concept(
        name_iast="tyaga",
        name_devanagari="त्याग",
        name_english="renunciation",
        description="Renunciation of the fruits of action; letting go",
        aliases_iast=["sannyasa", "vairagya"],
        related_concepts=["karma", "nishkamakarma", "yoga"],
    ),
    Concept(
        name_iast="samkhya",
        name_devanagari="सांख्य",
        name_english="knowledge/discrimination",
        description="Discriminative knowledge between Self and non-Self",
        aliases_iast=["sankhya", "buddhi", "viveka"],
        related_concepts=["jnana", "atman", "prakriti"],
    ),
    Concept(
        name_iast="yajna",
        name_devanagari="यज्ञ",
        name_english="sacrifice/offering",
        description="Sacrifice or offering; selfless action as worship",
        aliases_iast=["homa", "dana", "tapas"],
        related_concepts=["karma", "dharma", "ishvara"],
    ),
    Concept(
        name_iast="dhyana",
        name_devanagari="ध्यान",
        name_english="meditation",
        description="Meditation; focused contemplation on the divine",
        aliases_iast=["dharana", "samadhi", "yoga"],
        related_concepts=["yoga", "atman", "ishvara"],
    ),
    Concept(
        name_iast="sharira",
        name_devanagari="शरीर",
        name_english="body",
        description="The physical body; the three bodies (sthula, sukshma, karana)",
        aliases_iast=["deha", "tanu", "vapu"],
        related_concepts=["atman", "kshetra", "prakriti"],
    ),
    Concept(
        name_iast="sukha",
        name_devanagari="सुख",
        name_english="happiness",
        description="Happiness; pleasure; the nature of the Self is bliss",
        aliases_iast=["ananda", "priti"],
        related_concepts=["duhkha", "atman", "sattva"],
    ),
    Concept(
        name_iast="duhkha",
        name_devanagari="दुःख",
        name_english="suffering",
        description="Suffering; pain; the inherent unsatisfactoriness of worldly existence",
        aliases_iast=["soka", "moha", "bhaya"],
        related_concepts=["sukha", "samsara", "moksha"],
    ),
    Concept(
        name_iast="maya",
        name_devanagari="माया",
        name_english="illusion",
        description="The cosmic illusion; the power that conceals the true nature of reality",
        aliases_iast=["avidya", "tamas"],
        related_concepts=["prakriti", "samsara", "moksha"],
    ),
    Concept(
        name_iast="nishkamakarma",
        name_devanagari="निष्कामकर्म",
        name_english="desireless action",
        description="Action without attachment to results; selfless action",
        aliases_iast=["nishkama karma", "karma yoga"],
        related_concepts=["karma", "tyaga", "dharma"],
    ),
    Concept(
        name_iast="prapatti",
        name_devanagari="प्रपत्ति",
        name_english="surrender",
        description="Complete surrender to the divine; sharanagati",
        aliases_iast=["sharanagati", "atmanivedana"],
        related_concepts=["bhakti", "ishvara", "moksha"],
    ),
    Concept(
        name_iast="sthitaprajna",
        name_devanagari="स्थितप्रज्ञ",
        name_english="steadfast wisdom",
        description="One who is established in wisdom; the ideal sage",
        aliases_iast=["sthitaprajna", "gunatita"],
        related_concepts=["jnana", "yoga", "atman"],
    ),
    Concept(
        name_iast="vishada",
        name_devanagari="विषाद",
        name_english="dejection/despair",
        description="Moral crisis or despondency; the starting point of the Gita",
        aliases_iast=["soka", "moha", "kashmala"],
        related_concepts=["arjuna", "dharma", "yoga"],
    ),
]


class ConceptExtractor:
    """Extract philosophical concepts from Sanskrit text."""

    def __init__(self, concepts: list[Concept] = None):
        self.concepts = concepts or SEED_CONCEPTS
        self._build_index()

    def _build_index(self):
        """Build search index for concept matching."""
        self._name_to_concept: dict[str, Concept] = {}
        self._alias_to_concept: dict[str, Concept] = {}

        for concept in self.concepts:
            self._name_to_concept[concept.name_iast.lower()] = concept
            self._name_to_concept[concept.name_english.lower()] = concept

            for alias in concept.aliases_iast:
                self._alias_to_concept[alias.lower()] = concept

        logger.info(
            f"ConceptExtractor: {len(self.concepts)} concepts, "
            f"{len(self._name_to_concept)} names, "
            f"{len(self._alias_to_concept)} aliases"
        )

    def extract_from_text(self, text_iast: str) -> list[dict]:
        """Extract concepts from IAST text.

        Args:
            text_iast: Sanskrit text in IAST transliteration.

        Returns:
            List of dicts with concept info and match confidence.
        """
        text_lower = text_iast.lower()
        found_concepts = []

        for concept in self.concepts:
            confidence = 0.0

            if concept.name_iast.lower() in text_lower:
                confidence = 1.0
            elif concept.name_english.lower() in text_lower:
                confidence = 0.9
            else:
                for alias in concept.aliases_iast:
                    if alias.lower() in text_lower:
                        confidence = max(confidence, 0.8)
                        break

            if confidence > 0:
                found_concepts.append(
                    {
                        "concept": concept,
                        "confidence": confidence,
                        "matched_name": concept.name_iast,
                    }
                )

        found_concepts.sort(key=lambda x: x["confidence"], reverse=True)
        return found_concepts

    def extract_from_query(self, query: str) -> list[dict]:
        """Extract concepts from a user query (may be in any language).

        Args:
            query: User query string.

        Returns:
            List of dicts with concept info and match confidence.
        """
        return self.extract_from_text(query)

    def get_concept_by_name(self, name: str) -> Concept | None:
        """Look up a concept by name or alias."""
        name_lower = name.lower()
        concept = self._name_to_concept.get(name_lower)
        if concept:
            return concept
        return self._alias_to_concept.get(name_lower)

    def get_related_concepts(self, concept_name: str) -> list[Concept]:
        """Get concepts related to the given concept."""
        concept = self.get_concept_by_name(concept_name)
        if not concept:
            return []

        related = []
        for rel_name in concept.related_concepts:
            rel_concept = self.get_concept_by_name(rel_name)
            if rel_concept:
                related.append(rel_concept)
        return related

    def list_all_concepts(self) -> list[Concept]:
        """Return all available concepts."""
        return self.concepts.copy()
