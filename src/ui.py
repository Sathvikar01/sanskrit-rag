"""Gradio Web UI for SansRAG — Hybrid RAG System.

5 Tabs:
1. Ask a Question — Full pipeline with Docker-managed DBs, RRF reranking, SQL verse display
2. RRF Reranked Search — Cross-database retrieval with actual verse display from SQL
3. Verse-Level Graph Search — Neo4j multi-hop visualization
4. Semantic Search — Qdrant vector search display
5. System Stats — Database and component status
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import subprocess
import time
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import gradio as gr

from config.settings import (
    COLLECTION_NAMES,
    L1_REG_LAMBDA,
    L2_REG_LAMBDA,
    RRF_TOP_K,
)
from src.commentary_manager import COMMENTARY_CONFIG, CommentaryManager
from src.xml_parser import TEIXMLParser
from src.embedding_client import NVIDIAEmbeddingClient
from src.gemini_client import NVIDIA_LLM_Client
from src.qdrant_manager import QdrantManager
from src.neo4j_manager import Neo4jManager
from src.verse_db import EXPECTED_BHAGAVAD_GITA_VERSE_COUNT, VerseDatabase, ingest_xml_to_sqlite
from src.retriever import RegularizedRetriever, parse_verse_references, extract_sanskrit_lemmas
from src.answer_generator import AnswerGenerator, AnswerResult


def manage_docker(action: str) -> str:
    """Start, stop, or check Docker containers."""
    compose_file = str(ROOT_DIR / "docker-compose.yml")
    try:
        if action == "start":
            result = subprocess.run(
                ["docker", "compose", "-f", compose_file, "up", "-d"],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                return "Docker containers started (Qdrant + Neo4j). Waiting 10s for initialization..."
            return f"Failed to start Docker:\n{result.stderr}"
        elif action == "stop":
            result = subprocess.run(
                ["docker", "compose", "-f", compose_file, "down"],
                capture_output=True, text=True, timeout=30
            )
            return "Docker containers stopped." if result.returncode == 0 else f"Failed to stop:\n{result.stderr}"
        elif action == "status":
            result = subprocess.run(
                ["docker", "compose", "-f", compose_file, "ps"],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout if result.stdout.strip() else "No containers running."
    except FileNotFoundError:
        return "Docker not found. Please install Docker Desktop."
    except subprocess.TimeoutExpired:
        return "Docker command timed out."
    return ""


class SansRAGUI:
    """Web UI backend for SansRAG with split Neo4j/Qdrant retrieval and SQL verse display."""

    def __init__(self):
        self.parser = TEIXMLParser()
        self.embedder = NVIDIAEmbeddingClient()
        self.llm = NVIDIA_LLM_Client()
        self.qdrant = QdrantManager()
        self.neo4j = Neo4jManager()
        self.verse_db = VerseDatabase()
        self.commentary_manager = CommentaryManager(
            qdrant_manager=self.qdrant,
            embedding_client=self.embedder,
        )
        self.retriever = None
        self.answer_generator = None
        self._connected = False

    def _init_verse_db(self):
        """Initialize SQLite verse database from XML if not already populated."""
        self.verse_db.connect()
        stats = self.verse_db.get_stats()
        if stats["total_verses"] == 0 or stats["total_verses"] < EXPECTED_BHAGAVAD_GITA_VERSE_COUNT:
            xml_path = ROOT_DIR / "dataset.xml"
            if xml_path.exists():
                print("Ingesting verses from XML into SQLite...")
                ingest_xml_to_sqlite(str(xml_path), self.verse_db.db_path)
                self.verse_db.connect()
                stats = self.verse_db.get_stats()
                print(f"Verse DB ready: {stats['total_verses']} verses")
            else:
                print("WARNING: dataset.xml not found. SQL verse display will be empty.")

    def connect(self) -> bool:
        """Connect to Qdrant, Neo4j, and initialize verse DB."""
        if self._connected:
            return True

        try:
            qdrant_ok = self.qdrant.connect()
            neo4j_ok = self.neo4j.connect()
            self._init_verse_db()
            if qdrant_ok:
                self._init_commentary_store()

            self.retriever = RegularizedRetriever(
                embedding_client=self.embedder,
                qdrant_manager=self.qdrant if qdrant_ok else None,
                neo4j_manager=self.neo4j if neo4j_ok else None,
                l1_lambda=L1_REG_LAMBDA,
                l2_lambda=L2_REG_LAMBDA,
                adaptive=True,
                llm_client=self.llm,
            )
            self.retriever._neo4j_available = neo4j_ok
            self.retriever._qdrant_available = qdrant_ok
            self.answer_generator = AnswerGenerator(
                gemini_client=self.llm,
                retriever=self.retriever,
                qdrant_manager=self.qdrant if qdrant_ok else None,
                neo4j_manager=self.neo4j if neo4j_ok else None,
                verse_db=self.verse_db,
                top_k=RRF_TOP_K,
            )

            print("Pre-checking LLM quota...")
            if self.llm.pre_check_quota():
                print("LLM available for answer generation")
            else:
                print("LLM quota exhausted, answers will be context-only")

            self._connected = True
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def _init_commentary_store(self):
        """Ensure commentator collections are populated from raw XML."""
        xml_path = ROOT_DIR / "dataset.xml"
        if not xml_path.exists():
            return
        try:
            self.commentary_manager.ensure_commentary_ingested(
                str(xml_path),
                force_reingest=False,
                show_progress=False,
            )
        except Exception as exc:
            print(f"Commentary initialization warning: {exc}")

    def _format_verse_from_sql(self, verse: dict) -> str:
        """Format a verse dict from SQL into display markdown."""
        if not verse:
            return ""
        lines = verse.get("lines", [])

        md = f"### {verse['verse_id']}"
        if verse.get("speaker"):
            md += f" — *{verse['speaker']}*"
        md += "\n\n"
        if lines:
            md += "```\n" + "\n".join(lines) + "\n```\n"
        else:
            md += f"```\n{verse.get('sanskrit_text', '')[:500]}\n```\n"
        return md

    def _get_commentary_matches(self, query: str, verse_ids: list) -> list:
        """Retrieve the top commentary overall for each requested verse."""
        if not verse_ids:
            return []
        matches = self.commentary_manager.get_best_matches(query, verse_ids)
        return [match.to_dict() for match in matches]

    def _format_commentary_matches(
        self,
        commentary_matches: list,
        heading: str = "### Commentary Matches",
    ) -> str:
        """Format commentary matches for display."""
        if not commentary_matches:
            return "_No commentary match found for the retrieved verses._"

        parts = [heading]
        for match in commentary_matches:
            excerpt = (match.get("text", "") or "").replace("\n", " ").strip()
            if len(excerpt) > 500:
                excerpt = excerpt[:500].rstrip() + "..."

            metadata = match.get("metadata", {}) or {}
            meta_bits = []
            if match.get("score") is not None:
                meta_bits.append(f"Semantic score: {float(match['score']):.4f}")
            if metadata.get("source_dataset"):
                meta_bits.append(f"Source: {metadata['source_dataset']}")
            if metadata.get("text_variant"):
                meta_bits.append(f"Variant: {metadata['text_variant']}")

            parts.append(
                f"#### {match.get('verse_id', 'Unknown verse')} — "
                f"{match.get('author_display_name', match.get('author_key', 'Unknown'))}"
            )
            parts.append(f"> {excerpt}")
            if meta_bits:
                parts.append(f"*{' | '.join(meta_bits)}*")

        return "\n\n".join(parts)

    def _get_verses_from_sql(self, verse_ids: list) -> str:
        """Retrieve actual verses from SQL and format as markdown."""
        if not verse_ids:
            return "_No verses to display._"
        verses = self.verse_db.get_verses_by_ids(verse_ids)
        if not verses:
            return f"_Verse IDs not found in SQL database: {verse_ids[:5]}..._"
        parts = []
        for v in verses:
            parts.append(self._format_verse_from_sql(v))
        return "\n\n---\n\n".join(parts)

    def ask(
        self,
        query: str,
        top_k: int,
        l1_lambda: float,
        l2_lambda: float,
        regularization: str,
    ) -> tuple:
        """Execute full HybridRAG pipeline and return markdown answer + SQL verses."""
        if not query.strip():
            return "Please enter a search query.", "", "", "", "", ""

        if not self.connect():
            return "Failed to connect to databases.", "", "", "", "", ""

        self.retriever.l1_lambda = l1_lambda
        self.retriever.l2_lambda = l2_lambda
        self.answer_generator.top_k = int(top_k)

        try:
            result: AnswerResult = self.answer_generator.generate_answer(
                query,
                regularization=regularization,
            )
        except Exception as e:
            return f"Error: {str(e)}", "", "", "", "", ""

        verse_filter = parse_verse_references(query)
        db_status = result.sources.get("db_status", {})
        pipeline = f"**Pipeline:**\n"
        pipeline += f"- Language: {'English (no IAST)' if result.query == result.iast_query else 'Sanskrit (IAST applied)'}\n"
        pipeline += f"- Verse filter: {verse_filter.raw_match if verse_filter.has_filter() else 'None'}\n"
        pipeline += f"- Intent: {(result.query_intent or {}).get('intent', 'unknown')}\n"
        pipeline += f"- Retrieval: {result.retrieval_stats.get('retrieval_mode', 'hybrid')}\n"
        pipeline += f"- Candidates: {result.retrieval_stats.get('rrf_results', 0)}\n"
        pipeline += f"- Unique verses: {result.retrieval_stats.get('unique_verses', 0)}\n"
        pipeline += f"- Explicit refs: {', '.join(result.explicit_references) if result.explicit_references else 'None'}\n"
        pipeline += f"- Qdrant: {'contributed' if db_status.get('qdrant', {}).get('contributed') else 'not contributing'}\n"
        pipeline += f"- Neo4j: {'contributed' if db_status.get('neo4j', {}).get('contributed') else 'not contributing'}\n"
        pipeline += f"- Confidence: {result.confidence:.2f}\n"
        if result.abstention_reason:
            pipeline += f"- Abstention: {result.abstention_reason}\n"
        pipeline += f"- Cache: {json.dumps(result.cache, ensure_ascii=False)}\n"
        pipeline += f"- LLM re-rank: {'Yes' if self.llm.is_available() else 'No'}\n"
        pipeline += f"- Consistency: {result.consistency_score:.4f} (pass {result.retrieval_passes})\n"
        pipeline += f"- Latency: {result.latency_ms:.0f}ms"

        sources = (
            f"**Sources:** Qdrant={result.sources.get('qdrant_verses', 0)}"
            f" | Neo4j={result.sources.get('neo4j_verses', 0)}"
            f" | SQLite={result.sources.get('sqlite_verses', 0)}"
            f" | Graph={result.sources.get('graph_metadata', 0)}"
            f" | Commentary={len(result.commentary_matches)}"
        )

        citations_json = json.dumps(result.citations[:5], indent=2, ensure_ascii=False)

        sql_verse_ids = list(dict.fromkeys(
            v.get("verse_id", "")
            for v in result.evidence.get("canonical_verses", [])[:5]
            if v.get("verse_id")
        ))
        sql_verses_md = self._get_verses_from_sql(sql_verse_ids)
        commentary_md = self._format_commentary_matches(result.commentary_matches)

        return result.answer, pipeline, sources, citations_json, sql_verses_md, commentary_md

    def search_rrf(
        self,
        query: str,
        top_k: int,
        l1_lambda: float,
        l2_lambda: float,
        regularization: str,
    ) -> tuple:
        """Execute cross-database RRF reranked search + display actual verses from SQL."""
        if not query.strip():
            return "Please enter a search query.", "", "", ""

        if not self.connect():
            return "Failed to connect to databases.", "", "", ""

        self.retriever.l1_lambda = l1_lambda
        self.retriever.l2_lambda = l2_lambda

        verse_filter = parse_verse_references(query)
        try:
            results = self.retriever.cross_db_rrf_search(
                query,
                top_k=int(top_k),
                include_bm25=True,
                regularization=regularization,
                verse_filter=verse_filter,
            )
        except Exception as e:
            return f"Search error: {str(e)}", "", "", ""

        if not results:
            return "No results found.", "", "", "_No commentary match found for the retrieved verses._"

        output = []
        output.append(f"## RRF Results: `{query}`")
        if verse_filter.has_filter():
            output.append(f"**Verse filter:** {verse_filter.raw_match}")
        output.append(f"**Found:** {len(results)} results\n")

        for i, r in enumerate(results, 1):
            output.append(f"---")
            output.append(f"**#{i}** — `{r.verse_id or 'N/A'}` | **Score:** {r.final_score:.4f}")
            output.append(f"- Dense: {r.dense_score:.4f} | Sparse: {r.sparse_score:.4f} | BM25: {r.bm25_score:.4f}")
            src = r.metadata.get('sources', {})
            output.append(f"- Sources: Qdrant={src.get('qdrant', False)}, Neo4j={src.get('neo4j', False)}")
            if r.metadata.get('llm_rerank_score'):
                output.append(f"- LLM re-rank: {r.metadata['llm_rerank_score']}/5")
            output.append(f"\n{r.text[:300].replace(chr(10), ' ')}\n")

        verse_ids = []
        for result in results:
            if result.verse_id and result.verse_id not in verse_ids:
                verse_ids.append(result.verse_id)
            if len(verse_ids) >= 5 and not verse_filter.has_filter():
                break

        commentary_verse_ids = verse_filter.verse_ids if verse_filter.has_filter() else verse_ids
        commentary_matches = self._get_commentary_matches(query, commentary_verse_ids)

        json_out = {
            "query": query,
            "method": "cross_db_rrf",
            "verse_filter": verse_filter.to_dict() if verse_filter.has_filter() else None,
            "total_results": len(results),
            "results": [r.to_dict() for r in results],
            "commentary_matches": commentary_matches,
        }

        sql_verses_md = self._get_verses_from_sql([r.verse_id for r in results if r.verse_id])
        commentary_md = self._format_commentary_matches(commentary_matches)

        return "\n".join(output), json.dumps(json_out, indent=2, ensure_ascii=False), sql_verses_md, commentary_md

    def verse_graph_search(self, verse_id: str, top_k: int) -> tuple:
        """Search by verse ID with multi-hop graph traversal + SQL verse display."""
        if not self.connect():
            return "Failed to connect.", "", "", ""

        verse_id = verse_id.strip()
        if not verse_id.startswith("BhG"):
            verse_id = f"BhG {verse_id}"

        sql_verse = self.verse_db.get_verse(verse_id)
        sql_md = self._format_verse_from_sql(sql_verse) if sql_verse else f"_Verse `{verse_id}` not found in SQL database._"

        graph_md = f"### Verse: `{verse_id}`\n\n"
        all_results = []

        if self.neo4j:
            neo4j_results = self.neo4j.search_by_verse_id(verse_id)
            if neo4j_results:
                for r in neo4j_results:
                    lemmas = r.metadata.get("lemmas", [])
                    word_forms = r.metadata.get("word_forms", [])
                    graph_md += f"#### Text\n```\n{r.text}\n```\n\n"
                    graph_md += f"#### Lemmas ({len(lemmas)})\n"
                    graph_md += ", ".join(f"`{l}`" for l in lemmas[:30]) + "\n\n"
                    graph_md += f"#### Word Forms ({len(word_forms)})\n"
                    graph_md += ", ".join(f"`{w}`" for w in word_forms[:30]) + "\n\n"
                    graph_md += f"#### Graph Traversal\n"
                    graph_md += f"`Query` → `Lemma nodes` → `Word nodes` → `Chunk ({verse_id})`\n\n"
                    all_results.append(r)

            if not neo4j_results:
                lemmas = extract_sanskrit_lemmas(verse_id)
                if lemmas:
                    multi_hop = self.neo4j.search_multi_hop(lemmas, top_k=int(top_k))
                    graph_md += f"\n#### Multi-hop for lemmas: {lemmas}\n"
                    for r in multi_hop:
                        graph_md += f"- `{r.verse_id}` (score: {r.score:.4f}, matches: {r.metadata.get('direct_matches', 0)})\n"
                    all_results.extend(multi_hop)

        if self.qdrant:
            qdrant_results = self.qdrant.search_dense(
                self.embedder.embed_query(verse_id).dense_vector,
                top_k=int(top_k),
            )
            if qdrant_results:
                graph_md += f"\n#### Semantic matches (Qdrant)\n"
                for r in qdrant_results[:5]:
                    graph_md += f"- `{r.verse_id}` (cosine: {r.score:.4f})\n"

        json_out = json.dumps(
            [{"id": r.id, "verse_id": r.verse_id, "score": r.score, "text": r.text[:200]} for r in all_results[:10]],
            indent=2, ensure_ascii=False,
        )
        commentary_query = " ".join(sql_verse.get("lines", [])) if sql_verse else verse_id
        if sql_verse and not commentary_query.strip():
            commentary_query = sql_verse.get("sanskrit_text", verse_id)
        commentary_matches = self._get_commentary_matches(commentary_query, [verse_id])
        commentary_md = self._format_commentary_matches(commentary_matches, heading="### Commentary Match")
        return graph_md, json_out, sql_md, commentary_md

    def semantic_search(self, query: str, top_k: int) -> tuple:
        """Semantic search via Qdrant dense vectors."""
        if not self.connect():
            return "Failed to connect.", ""

        embedding = self.embedder.embed_query(query)
        results = self.qdrant.search_dense(
            embedding.dense_vector,
            top_k=int(top_k),
        )

        md = f"### Semantic Search: `{query}`\n\n"
        md += f"**Embedding dim:** {len(embedding.dense_vector)} | **Sparse features:** {len(embedding.sparse_vector)}\n\n"

        for i, r in enumerate(results, 1):
            md += f"**#{i}** — `{r.verse_id or 'N/A'}` | **Cosine:** {r.score:.4f}\n"
            md += f"{r.text[:300].replace(chr(10), ' ')}\n\n"

        json_out = json.dumps(
            [{"verse_id": r.verse_id, "score": r.score, "text": r.text[:200]} for r in results],
            indent=2, ensure_ascii=False,
        )
        return md, json_out

    def system_stats(self) -> str:
        """Get database and component statistics."""
        if not self.connect():
            return "Not connected."

        stats = []
        stats.append("### System Status\n")
        stats.append("| Component | Status |")
        stats.append("|---|---|")

        try:
            q_stats = self.qdrant.get_collection_stats()
            stats.append(f"| Qdrant (seg_lemma) | {q_stats.get('row_count', 'N/A'):,} points |")
            commentary_stats = self.commentary_manager.get_commentary_collection_stats()
            for author, config in COMMENTARY_CONFIG.items():
                stats.append(
                    f"| Qdrant Commentary ({config['display_name']}) | "
                    f"{commentary_stats.get(author, 0):,} points |"
                )
        except Exception:
            stats.append("| Qdrant (seg_lemma) | Not available |")

        try:
            n_stats = self.neo4j.get_collection_stats()
            stats.append(f"| Neo4j Chunks | {n_stats.get('chunk_count', 'N/A'):,} |")
            stats.append(f"| Neo4j Words | {n_stats.get('word_count', 'N/A'):,} |")
            stats.append(f"| Neo4j Lemmas | {n_stats.get('lemma_count', 'N/A'):,} |")
        except Exception:
            stats.append("| Neo4j | Not available |")

        try:
            v_stats = self.verse_db.get_stats()
            stats.append(f"| SQLite Verses | {v_stats['total_verses']:,} |")
            stats.append(f"| SQLite Chapters | {v_stats['chapters']} |")
        except Exception:
            stats.append("| SQLite (Verses) | Not available |")

        stats.append(f"| LLM (NVIDIA NIM) | {'Available' if self.llm.is_available() else 'Not available'} |")

        stats.append("\n### Architecture\n")
        stats.append("| Database | Data | Retrieval Mode |")
        stats.append("|---|---|---|")
        stats.append("| **Neo4j** | Lemma + Morphosyntax | Verse-level + Multi-hop graph |")
        stats.append("| **Qdrant** | Segmented + Lemmatized | Semantic (dense+sparse+BM25) |")
        stats.append("| **SQLite** | Raw Sanskrit verses | Direct verse lookup |")

        return "\n".join(stats)

    def docker_control(self, action: str) -> str:
        """Control Docker containers from the UI."""
        return manage_docker(action)


def create_ui():
    """Create and return Gradio interface."""
    # Lazy initialization to avoid startup blocking when services are down.
    _ui_instance = [None]

    def get_ui():
        """Lazy-initialize UI on first use."""
        if _ui_instance[0] is None:
            _ui_instance[0] = SansRAGUI()
        return _ui_instance[0]

    with gr.Blocks(title="SansRAG — Hybrid RAG System") as app:
        gr.Markdown("# SansRAG — Hybrid RAG System")
        gr.Markdown("**Neo4j** (verse-level graph + multi-hop) + **Qdrant** (semantic vectors) + **SQLite** (raw verses) with RRF reranking")

        with gr.Tabs():
            with gr.Tab("Ask a Question"):
                gr.Markdown("Full pipeline: Docker-managed DBs → language detection → verse parsing → dual retrieval → RRF reranking → LLM answer → SQL verse display")

                with gr.Row():
                    with gr.Column(scale=2):
                        query_input = gr.Textbox(
                            label="Question",
                            placeholder="e.g., What is the conchshell name of Yudhisthira? BG 1.16",
                            lines=2,
                        )
                        ask_btn = gr.Button("Generate Answer", variant="primary")

                    with gr.Column(scale=1):
                        top_k_slider = gr.Slider(1, 50, value=RRF_TOP_K, step=1, label="Top-K")
                        l1_slider = gr.Slider(0.0, 0.5, value=L1_REG_LAMBDA, step=0.001, label="L1 Lambda")
                        l2_slider = gr.Slider(0.0, 0.1, value=L2_REG_LAMBDA, step=0.0001, label="L2 Lambda")
                        reg_dropdown = gr.Dropdown(["combined", "l1", "l2", "none"], value="combined", label="Regularization")

                answer_md = gr.Markdown()

                with gr.Accordion("Pipeline Details", open=False):
                    pipeline_info = gr.Markdown()
                    sources_info = gr.Markdown()

                with gr.Accordion("Citations (JSON)", open=False):
                    citations_json = gr.Code(language="json", label="Citations")

                gr.Markdown("### Retrieved Verses (from SQL)")
                sql_verses_ask = gr.Markdown()
                gr.Markdown("### Commentary Matches")
                commentary_ask = gr.Markdown()

                ask_btn.click(
                    fn=lambda *args: get_ui().ask(*args),
                    inputs=[query_input, top_k_slider, l1_slider, l2_slider, reg_dropdown],
                    outputs=[answer_md, pipeline_info, sources_info, citations_json, sql_verses_ask, commentary_ask],
                )
                query_input.submit(
                    fn=lambda *args: get_ui().ask(*args),
                    inputs=[query_input, top_k_slider, l1_slider, l2_slider, reg_dropdown],
                    outputs=[answer_md, pipeline_info, sources_info, citations_json, sql_verses_ask, commentary_ask],
                )

            with gr.Tab("RRF Reranked Search"):
                gr.Markdown("Cross-database RRF fusion: Neo4j (verse-level) + Qdrant (semantic) → actual verses from SQL")

                with gr.Row():
                    with gr.Column(scale=2):
                        rrf_query = gr.Textbox(label="Search Query", placeholder="e.g., dharma kshetra", lines=1)
                        rrf_btn = gr.Button("Search", variant="primary")

                    with gr.Column(scale=1):
                        rrf_top_k = gr.Slider(1, 50, value=RRF_TOP_K, step=1, label="Top-K")
                        rrf_reg = gr.Dropdown(["combined", "l1", "l2", "none"], value="combined", label="Regularization")

                rrf_results_md = gr.Markdown()
                with gr.Accordion("Results (JSON)", open=False):
                    rrf_json = gr.Code(language="json", label="JSON")

                gr.Markdown("### Retrieved Verses (from SQL)")
                sql_verses_rrf = gr.Markdown()
                gr.Markdown("### Commentary Matches")
                commentary_rrf = gr.Markdown()

                rrf_btn.click(
                    fn=lambda *args: get_ui().search_rrf(*args),
                    inputs=[rrf_query, rrf_top_k, l1_slider, l2_slider, rrf_reg],
                    outputs=[rrf_results_md, rrf_json, sql_verses_rrf, commentary_rrf],
                )

            with gr.Tab("Verse-Level Graph Search"):
                gr.Markdown("Neo4j multi-hop traversal + actual verse from SQL database")

                with gr.Row():
                    with gr.Column(scale=2):
                        verse_input = gr.Textbox(label="Verse ID", placeholder="e.g., BhG 1.20 or 1.16-18", lines=1)
                        verse_btn = gr.Button("Search Verse", variant="primary")

                    with gr.Column(scale=1):
                        verse_top_k = gr.Slider(1, 50, value=10, step=1, label="Max Results")

                gr.Markdown("### Actual Verse (from SQL)")
                sql_verse_display = gr.Markdown()

                verse_md = gr.Markdown()
                with gr.Accordion("Results (JSON)", open=False):
                    verse_json = gr.Code(language="json", label="JSON")
                gr.Markdown("### Commentary Match")
                commentary_verse = gr.Markdown()

                verse_btn.click(
                    fn=lambda *args: get_ui().verse_graph_search(*args),
                    inputs=[verse_input, verse_top_k],
                    outputs=[verse_md, verse_json, sql_verse_display, commentary_verse],
                )

            with gr.Tab("Semantic Search"):
                gr.Markdown("Qdrant dense vector search for semantic relevance")

                with gr.Row():
                    with gr.Column(scale=2):
                        sem_query = gr.Textbox(label="Search Query", placeholder="e.g., duty and righteousness", lines=1)
                        sem_btn = gr.Button("Search", variant="primary")

                    with gr.Column(scale=1):
                        sem_top_k = gr.Slider(1, 50, value=10, step=1, label="Top-K")

                sem_results_md = gr.Markdown()
                with gr.Accordion("Results (JSON)", open=False):
                    sem_json = gr.Code(language="json", label="JSON")

                sem_btn.click(
                    fn=lambda *args: get_ui().semantic_search(*args),
                    inputs=[sem_query, sem_top_k],
                    outputs=[sem_results_md, sem_json],
                )

            with gr.Tab("System Stats"):
                gr.Markdown("### Docker Management")
                with gr.Row():
                    docker_start_btn = gr.Button("Start Docker (Qdrant + Neo4j)", variant="primary")
                    docker_stop_btn = gr.Button("Stop Docker", variant="stop")
                    docker_status_btn = gr.Button("Check Status")
                docker_output = gr.Textbox(label="Docker Output", lines=4)
                docker_start_btn.click(fn=lambda x: get_ui().docker_control(x), inputs=[gr.State("start")], outputs=[docker_output])
                docker_stop_btn.click(fn=lambda x: get_ui().docker_control(x), inputs=[gr.State("stop")], outputs=[docker_output])
                docker_status_btn.click(fn=lambda x: get_ui().docker_control(x), inputs=[gr.State("status")], outputs=[docker_output])

                stats_md = gr.Markdown()
                refresh_btn = gr.Button("Refresh Stats")
                refresh_btn.click(fn=lambda: get_ui().system_stats(), outputs=[stats_md])
                app.load(fn=lambda: get_ui().system_stats(), outputs=[stats_md])

        gr.Markdown("---\n**Tips:**\n- English questions skip IAST translation for better retrieval\n- Include verse references (e.g., BG 1.15) for precise verse-level lookup\n- The graph tab shows multi-hop lemma traversal paths\n- Verses are displayed from the local SQLite database")

    return app


def main():
    """Launch Gradio UI."""
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
