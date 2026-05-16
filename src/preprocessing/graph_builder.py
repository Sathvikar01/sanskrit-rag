"""Neo4j graph builder for loading Bhagavad Gita data."""

import json
from pathlib import Path

from neo4j import GraphDatabase

from src.utils.config import Config
from src.utils.logger import logger


class GraphBuilder:
    """Build and populate the Neo4j knowledge graph."""

    def __init__(self, config: Config = None):
        if config is None:
            config = Config()

        self.uri = config.neo4j_uri
        self.user = config.neo4j_user
        self.password = config.neo4j_password
        self.driver = None

    def connect(self):
        """Establish connection to Neo4j."""
        logger.info(f"Connecting to Neo4j at {self.uri}")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.driver.verify_connectivity()
        logger.info("Connected to Neo4j successfully")

    def close(self):
        """Close the Neo4j driver."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def clear_database(self):
        """Clear all nodes and relationships in the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared Neo4j database")

    def create_constraints(self):
        """Create uniqueness constraints and indexes."""
        constraints = [
            "CREATE CONSTRAINT chapter_num IF NOT EXISTS FOR (c:Chapter) REQUIRE c.number IS UNIQUE",
            "CREATE CONSTRAINT verse_ref IF NOT EXISTS FOR (v:Verse) REQUIRE v.ref IS UNIQUE",
            "CREATE CONSTRAINT commentary_id IF NOT EXISTS FOR (c:Commentary) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT commentator_id IF NOT EXISTS FOR (c:Commentator) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name_iast IS UNIQUE",
        ]

        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.warning(f"Constraint creation issue: {e}")

        logger.info("Created constraints and indexes")

    def create_fulltext_indexes(self):
        """Create full-text indexes for text search."""
        indexes = [
            """
            CREATE FULLTEXT INDEX verse_text_ft IF NOT EXISTS
            FOR (v:Verse) ON EACH [v.text_iast]
            """,
            """
            CREATE FULLTEXT INDEX commentary_text_ft IF NOT EXISTS
            FOR (c:Commentary) ON EACH [c.text_iast]
            """,
        ]

        with self.driver.session() as session:
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.warning(f"Index creation issue: {e}")

        logger.info("Created full-text indexes")

    def load_chapters(self, chapters_file: str | Path):
        """Load chapter nodes from JSON file."""
        with open(chapters_file, "r", encoding="utf-8") as f:
            chapters = json.load(f)

        query = """
        UNWIND $chapters AS ch
        MERGE (c:Chapter {number: ch.number})
        SET c.name_iast = ch.name_iast,
            c.name_devanagari = ch.name_devanagari,
            c.name_english = ch.name_english
        """

        with self.driver.session() as session:
            session.run(query, chapters=chapters)

        logger.info(f"Loaded {len(chapters)} chapters")

    def load_commentators(self, commentators_file: str | Path):
        """Load commentator nodes from JSON file."""
        with open(commentators_file, "r", encoding="utf-8") as f:
            commentators = json.load(f)

        query = """
        UNWIND $commentators AS cm
        MERGE (c:Commentator {id: cm.id})
        SET c.name_iast = cm.name_iast,
            c.name_devanagari = cm.name_devanagari,
            c.name_english = cm.name_english,
            c.tradition = cm.tradition,
            c.commentary_name = cm.commentary_name
        """

        with self.driver.session() as session:
            session.run(query, commentators=commentators)

        logger.info(f"Loaded {len(commentators)} commentators")

    def load_concepts(self, concepts_file: str | Path):
        """Load concept nodes from JSON file."""
        with open(concepts_file, "r", encoding="utf-8") as f:
            concepts = json.load(f)

        query = """
        UNWIND $concepts AS co
        MERGE (c:Concept {name_iast: co.name_iast})
        SET c.name_devanagari = co.name_devanagari,
            c.name_english = co.name_english,
            c.description = co.description
        """

        with self.driver.session() as session:
            session.run(query, concepts=concepts)

        logger.info(f"Loaded {len(concepts)} concepts")

    def load_verses(self, verses_file: str | Path):
        """Load verse nodes from JSON file."""
        with open(verses_file, "r", encoding="utf-8") as f:
            verses = json.load(f)

        query = """
        UNWIND $verses AS v
        MERGE (verse:Verse {ref: v.ref})
        SET verse.chapter_num = v.chapter_num,
            verse.verse_num = v.verse_num,
            verse.text_iast = v.text_iast,
            verse.text_devanagari = v.text_devanagari,
            verse.speaker = v.speaker,
            verse.lemmas = v.lemmas
        """

        with self.driver.session() as session:
            session.run(query, verses=verses)

        logger.info(f"Loaded {len(verses)} verses")

    def load_commentaries(self, commentaries_file: str | Path):
        """Load commentary nodes from JSON file."""
        with open(commentaries_file, "r", encoding="utf-8") as f:
            commentaries = json.load(f)

        query = """
        UNWIND $commentaries AS cm
        MERGE (c:Commentary {id: cm.id})
        SET c.verse_ref = cm.verse_ref,
            c.commentator = cm.commentator,
            c.text_iast = cm.text_iast,
            c.text_devanagari = cm.text_devanagari,
            c.lemmas = cm.lemmas
        """

        with self.driver.session() as session:
            session.run(query, commentaries=commentaries)

        logger.info(f"Loaded {len(commentaries)} commentaries")

    def load_relationships(self, relationships_file: str | Path):
        """Load relationships from JSON file."""
        with open(relationships_file, "r", encoding="utf-8") as f:
            relationships = json.load(f)

        by_type = {}
        for rel in relationships:
            rel_type = rel["rel_type"]
            if rel_type not in by_type:
                by_type[rel_type] = []
            by_type[rel_type].append(rel)

        with self.driver.session() as session:
            for rel_type, rels in by_type.items():
                if rel_type == "IN_CHAPTER":
                    query = """
                    UNWIND $rels AS r
                    MATCH (v:Verse {ref: r.start_value})
                    MATCH (c:Chapter {number: toInteger(r.end_value)})
                    MERGE (v)-[:IN_CHAPTER]->(c)
                    """
                elif rel_type == "HAS_COMMENTARY":
                    query = """
                    UNWIND $rels AS r
                    MATCH (v:Verse {ref: r.start_value})
                    MATCH (c:Commentary {id: r.end_value})
                    MERGE (v)-[:HAS_COMMENTARY]->(c)
                    """
                elif rel_type == "BY_COMMENTATOR":
                    query = """
                    UNWIND $rels AS r
                    MATCH (c:Commentary {id: r.start_value})
                    MATCH (cm:Commentator {id: r.end_value})
                    MERGE (c)-[:BY_COMMENTATOR]->(cm)
                    """
                elif rel_type == "MENTIONS_CONCEPT":
                    query = """
                    UNWIND $rels AS r
                    MATCH (v:Verse {ref: r.start_value})
                    MATCH (c:Concept {name_iast: r.end_value})
                    MERGE (v)-[rel:MENTIONS_CONCEPT]->(c)
                    SET rel.confidence = r.properties.confidence
                    """
                elif rel_type == "NEXT_VERSE":
                    query = """
                    UNWIND $rels AS r
                    MATCH (v1:Verse {ref: r.start_value})
                    MATCH (v2:Verse {ref: r.end_value})
                    MERGE (v1)-[:NEXT_VERSE]->(v2)
                    """
                elif rel_type == "RELATED_TO":
                    query = """
                    UNWIND $rels AS r
                    MATCH (c1:Concept {name_iast: r.start_value})
                    MATCH (c2:Concept {name_iast: r.end_value})
                    MERGE (c1)-[:RELATED_TO]->(c2)
                    """
                else:
                    logger.warning(f"Unknown relationship type: {rel_type}")
                    continue

                session.run(query, rels=rels)
                logger.info(f"Loaded {len(rels)} {rel_type} relationships")

        logger.info(f"Loaded {len(relationships)} total relationships")

    def build_from_files(self, import_dir: str | Path, clear: bool = True):
        """Build the complete graph from import JSON files.

        Args:
            import_dir: Directory containing the JSON import files.
            clear: Whether to clear the database first.
        """
        import_dir = Path(import_dir)

        if clear:
            self.clear_database()

        self.create_constraints()
        self.create_fulltext_indexes()

        self.load_chapters(import_dir / "chapters.json")
        self.load_commentators(import_dir / "commentators.json")
        self.load_concepts(import_dir / "concepts.json")
        self.load_verses(import_dir / "verses.json")
        self.load_commentaries(import_dir / "commentaries.json")
        self.load_relationships(import_dir / "relationships.json")

        logger.info("Graph construction complete!")

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self.driver.session() as session:
            result = session.run("""
                CALL apoc.meta.stats() YIELD nodeCount, relCount, labels, relTypes
                RETURN nodeCount, relCount, labels, relTypes
            """)
            record = result.single()
            if record:
                return {
                    "node_count": record["nodeCount"],
                    "relationship_count": record["relCount"],
                    "labels": dict(record["labels"]),
                    "rel_types": dict(record["relTypes"]),
                }

        with self.driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]

            labels = {}
            for record in session.run("CALL db.labels()"):
                label = record["label"]
                count = session.run(f"MATCH (n:`{label}`) RETURN count(n) as count").single()["count"]
                labels[label] = count

            return {
                "node_count": node_count,
                "relationship_count": rel_count,
                "labels": labels,
            }
