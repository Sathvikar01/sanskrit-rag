"""Curated Bhagavad Gita entity aliases for query expansion and evidence labels."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class EntityMatch:
    """A detected Gita entity in a user query."""

    key: str
    canonical: str
    matched_alias: str
    aliases: List[str]
    description: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "canonical": self.canonical,
            "matched_alias": self.matched_alias,
            "aliases": self.aliases,
            "description": self.description,
        }


ENTITY_LEXICON: Dict[str, Dict[str, object]] = {
    "krishna": {
        "canonical": "Krishna",
        "aliases": [
            "krishna",
            "krsna",
            "kṛṣṇa",
            "kesava",
            "keśava",
            "madhava",
            "mādhava",
            "govinda",
            "vasudeva",
            "vāsudeva",
            "janardana",
            "janārdana",
            "hrishikesha",
            "hṛṣīkeśa",
            "achyuta",
            "acyuta",
            "partha-sarathi",
            "pārtha-sārathi",
            "yashoda-nandana",
            "yaśodā-nandana",
        ],
        "description": "The speaker of the Gita and Arjuna's charioteer.",
    },
    "arjuna": {
        "canonical": "Arjuna",
        "aliases": [
            "arjuna",
            "partha",
            "pārtha",
            "dhananjaya",
            "dhanañjaya",
            "gudakesha",
            "guḍākeśa",
            "kaunteya",
            "pandava",
            "pāṇḍava",
            "savyasachi",
            "savyasācī",
        ],
        "description": "The warrior-disciple receiving Krishna's teaching.",
    },
    "dhritarashtra": {
        "canonical": "Dhritarashtra",
        "aliases": ["dhritarashtra", "dhṛtarāṣṭra", "dhrtarastra", "dhrtarashtra"],
        "description": "The blind Kuru king who asks Sanjaya about the battlefield.",
    },
    "sanjaya": {
        "canonical": "Sanjaya",
        "aliases": ["sanjaya", "sañjaya", "sanjay"],
        "description": "The narrator reporting the battlefield events to Dhritarashtra.",
    },
    "kurukshetra": {
        "canonical": "Kurukshetra",
        "aliases": ["kurukshetra", "kurukṣetra", "kuru-kshetra", "kuru-kṣetra", "dharma-kshetra", "dharma-kṣetra"],
        "description": "The battlefield described as a field of dharma.",
    },
    "dharma": {
        "canonical": "Dharma",
        "aliases": ["dharma", "svadharma", "sva-dharma", "righteousness", "duty"],
        "description": "Duty, righteousness, or sustaining order, depending on context.",
    },
    "karma": {
        "canonical": "Karma",
        "aliases": ["karma", "karman", "action", "work", "deed"],
        "description": "Action and its moral or spiritual relation to duty.",
    },
    "yoga": {
        "canonical": "Yoga",
        "aliases": ["yoga", "discipline", "spiritual practice"],
        "description": "A disciplined means of spiritual realization.",
    },
    "atman": {
        "canonical": "Atman",
        "aliases": ["atman", "ātman", "self", "soul"],
        "description": "The self or soul discussed in the Gita.",
    },
    "bhakti": {
        "canonical": "Bhakti",
        "aliases": ["bhakti", "devotion", "loving devotion"],
        "description": "Devotional orientation toward the divine.",
    },
    "yudhisthira": {
        "canonical": "Yudhishthira",
        "aliases": ["yudhisthira", "yudhiṣṭhira", "dharmaraja", "dharmarāja"],
        "description": "The eldest Pandava.",
    },
    "nakula": {
        "canonical": "Nakula",
        "aliases": ["nakula"],
        "description": "One of the Pandava brothers.",
    },
    "sahadeva": {
        "canonical": "Sahadeva",
        "aliases": ["sahadeva"],
        "description": "One of the Pandava brothers.",
    },
    "hanuman": {
        "canonical": "Hanuman",
        "aliases": ["hanuman", "hanumān"],
        "description": "The emblem on Arjuna's chariot flag in Chapter 1.",
    },
}


def _ascii_fold(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text or "")
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = folded.lower()
    folded = re.sub(r"[^a-z0-9\s.-]", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def find_entities(query: str) -> List[EntityMatch]:
    """Find known Gita entities by canonical names or aliases."""
    folded_query = f" {_ascii_fold(query)} "
    matches: List[EntityMatch] = []
    seen = set()

    for key, config in ENTITY_LEXICON.items():
        aliases = list(config["aliases"])
        for alias in aliases:
            folded_alias = _ascii_fold(alias)
            if not folded_alias:
                continue
            pattern = rf"(?<![a-z0-9]){re.escape(folded_alias)}(?![a-z0-9])"
            if re.search(pattern, folded_query):
                if key not in seen:
                    seen.add(key)
                    matches.append(
                        EntityMatch(
                            key=key,
                            canonical=str(config["canonical"]),
                            matched_alias=alias,
                            aliases=aliases,
                            description=str(config["description"]),
                        )
                    )
                break

    return matches


def expand_query_with_aliases(query: str, max_aliases_per_entity: int = 5) -> Dict[str, object]:
    """Return a retrieval-oriented query expansion while preserving the original query."""
    entities = find_entities(query)
    alias_terms: List[str] = []
    for entity in entities:
        for alias in entity.aliases[:max_aliases_per_entity]:
            folded_alias = _ascii_fold(alias)
            if folded_alias and folded_alias not in _ascii_fold(query):
                alias_terms.append(alias)

    unique_aliases = list(dict.fromkeys(alias_terms))
    expanded_query = query
    if unique_aliases:
        expanded_query = f"{query}\nRelated aliases: {', '.join(unique_aliases[:24])}"

    return {
        "original_query": query,
        "expanded_query": expanded_query,
        "entities": [entity.to_dict() for entity in entities],
        "aliases_added": unique_aliases[:24],
    }
