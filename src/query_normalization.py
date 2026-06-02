"""Semantic query normalization and Sanskrit/domain expansion helpers."""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List


SANSKRIT_QUERY_EXPANSIONS: Dict[str, List[str]] = {
    "arjuna": ["partha", "kaunteya", "gudakesa", "pandava", "gandiva"],
    "partha": ["arjuna", "son of prtha", "kunti"],
    "krsna": ["krishna", "vasudeva", "madhava", "govinda", "hrsikesa", "kesava"],
    "krishna": ["krsna", "vasudeva", "madhava", "govinda", "hrsikesa", "kesava"],
    "vasudeva": ["krsna", "krishna", "son of vasudeva"],
    "yashoda": ["yasoda", "yashoda nandana", "childhood pastimes", "vrndavana"],
    "yasoda": ["yashoda", "yashoda nandana", "childhood pastimes", "vrndavana"],
    "partha sarathi": ["arjuna charioteer", "krsna charioteer", "friend arjuna"],
    "gudakesa": ["conqueror of sleep", "arjuna"],
    "keshi": ["kesi", "killer of kesi demon", "krsna"],
    "kesi": ["keshi", "killer of kesi demon", "krsna"],
    "gandiva": ["arjuna bow", "dhanus", "bow"],
    "conchshell": ["conch", "pancajanya", "devadatta", "anantavijaya", "sughosa", "manipuspaka"],
    "conch": ["conchshell", "pancajanya", "devadatta", "anantavijaya", "sughosa", "manipuspaka"],
    "yudhisthira": ["anantavijaya", "king yudhisthira"],
    "nakula": ["sughosa"],
    "sahadeva": ["manipuspaka"],
    "hanuman": ["flag", "dhvaja", "rama"],
    "sita": ["goddess of fortune", "rama"],
    "dhrtarastra": ["dhritarashtra", "sons of dhrtarastra", "kaurava"],
    "dhritarashtra": ["dhrtarastra", "sons of dhrtarastra", "kaurava"],
    "duryodhana": ["obstinacy", "peaceful negotiation", "kaurava"],
    "bhisma": ["kuru chief", "grandsire"],
    "drona": ["teacher", "acharya"],
    "ksatriya": ["kshatriya", "warrior", "battle", "fighting"],
    "kshatriya": ["ksatriya", "warrior", "battle", "fighting"],
    "varnasrama": ["sanatana dharma", "atonement", "family tradition", "vishnu"],
    "sanatana": ["varnasrama", "dharma", "vishnu"],
    "prayaschitta": ["atonement", "sinful activities"],
    "prayascitta": ["prayaschitta", "atonement", "sinful activities"],
    "visnu": ["vishnu", "krsna", "self interest"],
    "vishnu": ["visnu", "krsna", "self interest"],
    "aggressors": ["atatayinah", "six aggressors", "poison giver", "arsonist"],
    "family": ["kula", "elders", "women", "children", "family tradition"],
    "forefathers": ["pitr", "food and water", "offerings"],
    "food": ["vishnu remnants", "sinful reactions"],
    "grief": ["compassion", "distress", "bow aside", "arjuna"],
    "distressed": ["grief", "compassion", "kinsmen", "arjuna"],
}


def _fold(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text or "")
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = folded.replace("ṛ", "r").replace("ṝ", "r")
    folded = folded.replace("ṣ", "s").replace("ś", "s")
    folded = folded.replace("ṭ", "t").replace("ḍ", "d").replace("ṇ", "n")
    folded = folded.replace("ṅ", "n").replace("ñ", "n").replace("ṃ", "m").replace("ḥ", "h")
    folded = folded.replace("-", " ").replace("_", " ")
    folded = re.sub(r"[^a-zA-Z0-9\s]", " ", folded)
    return re.sub(r"\s+", " ", folded).strip().lower()


def _ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def expand_semantic_query(query: str) -> str:
    """Append domain synonyms for Sanskrit Chapter 1 semantic retrieval."""
    folded_query = _fold(query)
    expansions: List[str] = []
    for trigger, values in SANSKRIT_QUERY_EXPANSIONS.items():
        if re.search(rf"\b{re.escape(trigger)}\b", folded_query):
            expansions.extend(values)

    expansions = _ordered_unique(expansions)
    if not expansions:
        return query
    return f"{query} {' '.join(expansions)}"
