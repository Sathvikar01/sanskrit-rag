"""Generate Neo4j import data from parsed Bhagavad Gita data."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.preprocessing.chunker import Chunk
from src.preprocessing.concept_extractor import Concept, ConceptExtractor, SEED_CONCEPTS
from src.preprocessing.iast_devanagari import get_converter
from src.preprocessing.xml_parser import VerseData, parse_verse_range
from src.utils.logger import logger


@dataclass
class GraphNode:
    """A node for Neo4j import."""

    label: str
    properties: dict


@dataclass
class GraphRelationship:
    """A relationship for Neo4j import."""

    start_label: str
    start_key: str
    start_value: str
    end_label: str
    end_key: str
    end_value: str
    rel_type: str
    properties: dict = field(default_factory=dict)


COMMENTATORS = {
    "sridhara": {
        "name_iast": "śrīdhara-svāmin",
        "name_devanagari": "श्रीधरस्वामिन्",
        "name_english": "Sridhara Swamin",
        "tradition": "advaita",
        "commentary_name": "gītā-subodhinī",
    },
    "visvanatha": {
        "name_iast": "viśvanātha-cakravartin",
        "name_devanagari": "विश्वनाथचक्रवर्तिन्",
        "name_english": "Visvanatha Chakravarti",
        "tradition": "acintya-bhedabheda",
        "commentary_name": "",
    },
    "baladeva": {
        "name_iast": "baladeva-vidyābhūṣaṇa",
        "name_devanagari": "बलदेवविद्याभूषण",
        "name_english": "Baladeva Vidyabhushana",
        "tradition": "acintya-bhedabheda",
        "commentary_name": "gītā-bhūṣaṇa",
    },
}

CHAPTER_NAMES = {
    1: ("arjuna-viṣāda-yoga", "अर्जुनविषादयोगः", "The Yoga of Arjuna's Dejection"),
    2: ("sāṅkhya-yoga", "सांख्ययोगः", "The Yoga of Knowledge"),
    3: ("karma-yoga", "कर्मयोगः", "The Yoga of Action"),
    4: ("jñāna-karma-sannyāsa-yoga", "ज्ञानकर्मसंन्यासयोगः", "The Yoga of Knowledge and Renunciation"),
    5: ("karma-sannyāsa-yoga", "कर्मसंन्यासयोगः", "The Yoga of Renunciation"),
    6: ("dhyāna-yoga", "ध्यानयोगः", "The Yoga of Meditation"),
    7: ("jñāna-vijñāna-yoga", "ज्ञानविज्ञानयोगः", "The Yoga of Knowledge and Wisdom"),
    8: ("akṣara-brahma-yoga", "अक्षरब्रह्मयोगः", "The Yoga of the Imperishable Absolute"),
    9: ("rāja-vidyā-rāja-guhya-yoga", "राजविद्याराजगुह्ययोगः", "The Yoga of Royal Knowledge"),
    10: ("vibhūti-yoga", "विभूतियोगः", "The Yoga of Divine Glories"),
    11: ("viśvarūpa-darśana-yoga", "विश्वरूपदर्शनयोगः", "The Yoga of the Universal Form"),
    12: ("bhakti-yoga", "भक्तियोगः", "The Yoga of Devotion"),
    13: ("kṣetra-kṣetrajña-vibhāga-yoga", "क्षेत्रक्षेत्रज्ञविभागयोगः", "The Yoga of the Field and the Knower"),
    14: ("guṇatraya-vibhāga-yoga", "गुणत्रयविभागयोगः", "The Yoga of the Three Gunas"),
    15: ("puruṣottama-yoga", "पुरुषोत्तमयोगः", "The Yoga of the Supreme Person"),
    16: ("daivāsura-sampad-vibhāga-yoga", "दैवासुरसम्पद्विभागयोगः", "The Yoga of Divine and Demoniac Natures"),
    17: ("śraddhātraya-vibhāga-yoga", "श्रद्धात्रयविभागयोगः", "The Yoga of the Threefold Faith"),
    18: ("mokṣa-sannyāsa-yoga", "मोक्षसंन्यासयोगः", "The Yoga of Liberation and Renunciation"),
}


def generate_chapter_nodes() -> list[GraphNode]:
    """Generate Chapter nodes for Neo4j."""
    nodes = []
    for num, (name_iast, name_deva, name_eng) in CHAPTER_NAMES.items():
        nodes.append(
            GraphNode(
                label="Chapter",
                properties={
                    "number": num,
                    "name_iast": name_iast,
                    "name_devanagari": name_deva,
                    "name_english": name_eng,
                },
            )
        )
    return nodes


def generate_commentator_nodes() -> list[GraphNode]:
    """Generate Commentator nodes for Neo4j."""
    nodes = []
    for key, info in COMMENTATORS.items():
        nodes.append(
            GraphNode(
                label="Commentator",
                properties={
                    "id": key,
                    "name_iast": info["name_iast"],
                    "name_devanagari": info["name_devanagari"],
                    "name_english": info["name_english"],
                    "tradition": info["tradition"],
                    "commentary_name": info["commentary_name"],
                },
            )
        )
    return nodes


def generate_concept_nodes(concepts: list[Concept] = None) -> list[GraphNode]:
    """Generate Concept nodes for Neo4j."""
    if concepts is None:
        concepts = SEED_CONCEPTS

    nodes = []
    for concept in concepts:
        nodes.append(
            GraphNode(
                label="Concept",
                properties={
                    "name_iast": concept.name_iast,
                    "name_devanagari": concept.name_devanagari,
                    "name_english": concept.name_english,
                    "description": concept.description,
                },
            )
        )
    return nodes


def generate_verse_nodes(
    verses: list[VerseData],
    chunks: list[Chunk],
) -> list[GraphNode]:
    """Generate Verse nodes for Neo4j. Expands range refs into individual Verse nodes."""
    converter = get_converter()
    chunk_map = {c.verse_ref: c for c in chunks if c.chunk_type == "verse"}

    nodes = []
    for verse in verses:
        text_iast = " ".join(verse.verse_lines_iast)
        text_deva = converter.iast_to_devanagari(text_iast)
        ch, lo, hi = parse_verse_range(verse.ref)

        for v in range(lo, hi + 1):
            individual_ref = f"BhG {ch}.{v}"
            chunk = chunk_map.get(individual_ref)
            lemmas = chunk.lemmas if chunk else []

            nodes.append(
                GraphNode(
                    label="Verse",
                    properties={
                        "ref": individual_ref,
                        "chapter_num": ch,
                        "verse_num": v,
                        "text_iast": text_iast,
                        "text_devanagari": text_deva,
                        "speaker": verse.speaker,
                        "lemmas": lemmas,
                    },
                )
            )
    return nodes


def generate_commentary_nodes(
    verses: list[VerseData],
    chunks: list[Chunk],
) -> list[GraphNode]:
    """Generate Commentary nodes for Neo4j. Expands range refs into individual nodes."""
    converter = get_converter()

    nodes = []
    for verse in verses:
        ch, lo, hi = parse_verse_range(verse.ref)

        for comm_key in ["sridhara", "visvanatha", "baladeva"]:
            lines = getattr(verse, f"{comm_key}_lines", [])
            if not lines:
                continue

            text_iast = " ".join(lines)
            text_deva = converter.iast_to_devanagari(text_iast)

            for v in range(lo, hi + 1):
                individual_ref = f"BhG {ch}.{v}"
                chunk_id = f"{individual_ref.replace(' ', '_')}_{comm_key}"
                chunk = next((c for c in chunks if c.chunk_id == chunk_id), None)
                lemmas = chunk.lemmas if chunk else []

                nodes.append(
                    GraphNode(
                        label="Commentary",
                        properties={
                            "id": chunk_id,
                            "verse_ref": individual_ref,
                            "commentator": comm_key,
                            "text_iast": text_iast[:5000],
                            "text_devanagari": text_deva[:5000],
                            "lemmas": lemmas,
                        },
                    )
                )
    return nodes


def generate_relationships(
    verses: list[VerseData],
    concepts: list[Concept] = None,
    concept_extractor: ConceptExtractor = None,
) -> list[GraphRelationship]:
    """Generate relationships for Neo4j import. Expands range refs."""
    if concept_extractor is None:
        concept_extractor = ConceptExtractor()

    relationships = []
    all_individual_refs = []

    for verse in verses:
        ch, lo, hi = parse_verse_range(verse.ref)

        for v in range(lo, hi + 1):
            individual_ref = f"BhG {ch}.{v}"
            all_individual_refs.append((individual_ref, ch, v))

            relationships.append(
                GraphRelationship(
                    start_label="Verse",
                    start_key="ref",
                    start_value=individual_ref,
                    end_label="Chapter",
                    end_key="number",
                    end_value=str(ch),
                    rel_type="IN_CHAPTER",
                )
            )

            for comm_key in ["sridhara", "visvanatha", "baladeva"]:
                lines = getattr(verse, f"{comm_key}_lines", [])
                if lines:
                    chunk_id = f"{individual_ref.replace(' ', '_')}_{comm_key}"
                    relationships.append(
                        GraphRelationship(
                            start_label="Verse",
                            start_key="ref",
                            start_value=individual_ref,
                            end_label="Commentary",
                            end_key="id",
                            end_value=chunk_id,
                            rel_type="HAS_COMMENTARY",
                        )
                    )
                    relationships.append(
                        GraphRelationship(
                            start_label="Commentary",
                            start_key="id",
                            start_value=chunk_id,
                            end_label="Commentator",
                            end_key="id",
                            end_value=comm_key,
                            rel_type="BY_COMMENTATOR",
                        )
                    )

            text_iast = " ".join(verse.verse_lines_iast)
            found_concepts = concept_extractor.extract_from_text(text_iast)
            for fc in found_concepts:
                concept = fc["concept"]
                relationships.append(
                    GraphRelationship(
                        start_label="Verse",
                        start_key="ref",
                        start_value=individual_ref,
                        end_label="Concept",
                        end_key="name_iast",
                        end_value=concept.name_iast,
                        rel_type="MENTIONS_CONCEPT",
                        properties={"confidence": fc["confidence"]},
                    )
                )

    for i in range(len(all_individual_refs) - 1):
        ref_a, ch_a, v_a = all_individual_refs[i]
        ref_b, ch_b, v_b = all_individual_refs[i + 1]
        if ch_a == ch_b:
            relationships.append(
                GraphRelationship(
                    start_label="Verse",
                    start_key="ref",
                    start_value=ref_a,
                    end_label="Verse",
                    end_key="ref",
                    end_value=ref_b,
                    rel_type="NEXT_VERSE",
                )
            )

    concept_map = {c.name_iast: c for c in (concepts or SEED_CONCEPTS)}
    for concept in (concepts or SEED_CONCEPTS):
        for rel_name in concept.related_concepts:
            if rel_name in concept_map:
                relationships.append(
                    GraphRelationship(
                        start_label="Concept",
                        start_key="name_iast",
                        start_value=concept.name_iast,
                        end_label="Concept",
                        end_key="name_iast",
                        end_value=rel_name,
                        rel_type="RELATED_TO",
                    )
                )

    logger.info(f"Generated {len(relationships)} relationships")
    return relationships


def save_graph_import_data(
    output_dir: str | Path,
    verses: list[VerseData],
    chunks: list[Chunk],
    concepts: list[Concept] = None,
) -> None:
    """Save all graph import data as JSON files.

    Args:
        output_dir: Directory to save JSON files.
        verses: Parsed verse data.
        chunks: Created chunks.
        concepts: Optional concept list (uses SEED_CONCEPTS if None).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    concept_extractor = ConceptExtractor(concepts)

    chapter_nodes = generate_chapter_nodes()
    commentator_nodes = generate_commentator_nodes()
    concept_nodes = generate_concept_nodes(concepts)
    verse_nodes = generate_verse_nodes(verses, chunks)
    commentary_nodes = generate_commentary_nodes(verses, chunks)
    relationships = generate_relationships(verses, concepts, concept_extractor)

    def save_nodes(nodes: list[GraphNode], filename: str):
        filepath = output_dir / filename
        data = [n.properties for n in nodes]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(data)} nodes to {filepath}")

    save_nodes(chapter_nodes, "chapters.json")
    save_nodes(commentator_nodes, "commentators.json")
    save_nodes(concept_nodes, "concepts.json")
    save_nodes(verse_nodes, "verses.json")
    save_nodes(commentary_nodes, "commentaries.json")

    rel_filepath = output_dir / "relationships.json"
    rel_data = []
    for r in relationships:
        rel_data.append(
            {
                "start_label": r.start_label,
                "start_key": r.start_key,
                "start_value": r.start_value,
                "end_label": r.end_label,
                "end_key": r.end_key,
                "end_value": r.end_value,
                "rel_type": r.rel_type,
                "properties": r.properties,
            }
        )
    with open(rel_filepath, "w", encoding="utf-8") as f:
        json.dump(rel_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(rel_data)} relationships to {rel_filepath}")
