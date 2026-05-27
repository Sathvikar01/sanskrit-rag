"""Hybrid Retrieval System with Verse-Level Graph + Semantic Vector Search."""
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    COLLECTION_NAMES,
    L1_REG_LAMBDA,
    L2_REG_LAMBDA,
    RRF_K,
    MIN_RRF_SCORE,
    RRF_WEIGHTS_VERSE_FILTER,
    RRF_WEIGHTS_NO_FILTER,
)
from src.embedding_client import NVIDIAEmbeddingClient, EmbeddingResult, apply_regularization
from src.qdrant_manager import QdrantManager, SearchResult as QdrantSearchResult
from src.neo4j_manager import Neo4jManager, SearchResult as Neo4jSearchResult


@dataclass
class VerseFilter:
    """Parsed verse reference from a query."""
    chapter: Optional[int] = None
    verse_start: Optional[int] = None
    verse_end: Optional[int] = None
    verse_ids: Optional[List[str]] = None
    raw_match: str = ""

    def has_filter(self) -> bool:
        return self.chapter is not None or (self.verse_ids is not None and len(self.verse_ids) > 0)

    def to_qdrant_dict(self) -> Dict[str, Any]:
        result = {}
        if self.verse_ids:
            result["verse_ids"] = self.verse_ids
        if self.chapter:
            result["chapter"] = self.chapter
            result["verse_start"] = self.verse_start or 1
            result["verse_end"] = self.verse_end or self.verse_start or 1
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chapter": self.chapter,
            "verse_start": self.verse_start,
            "verse_end": self.verse_end,
            "verse_ids": self.verse_ids,
            "raw_match": self.raw_match,
        }


def parse_verse_references(query: str) -> VerseFilter:
    """Extract verse references from a query.

    Matches patterns like:
    - BG 1.15, BhG 1.15, bg 1.15
    - BG 1.16-18, BhG 1.20-25
    - 1.15, 1.16-18
    - Bhagavad Gita 1.15, Chapter 1 Verse 15
    """
    patterns = [
        r'(?:BG|BhG|Bhagavad\s*Gita)\.?\s*(\d+)\.(\d+)(?:-(\d+))?',
        r'(?i)chapter\s*(\d+)\s*(?:verse\s*|v\s*)?(\d+)(?:-(\d+))?',
        r'(?<!\w)(\d+)\.(\d+)(?:-(\d+))?(?!\w)',
    ]

    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            chapter = int(match.group(1))
            verse_start = int(match.group(2))
            verse_end = int(match.group(3)) if match.group(3) else verse_start

            verse_ids = [f"BhG {chapter}.{v}" for v in range(verse_start, verse_end + 1)]

            return VerseFilter(
                chapter=chapter,
                verse_start=verse_start,
                verse_end=verse_end,
                verse_ids=verse_ids,
                raw_match=match.group(0)
            )

    return VerseFilter()


def extract_sanskrit_lemmas(query: str) -> List[str]:
    """Extract potential Sanskrit lemmas/terms from a query.
    
    Looks for IAST words, known Sanskrit terms, and capitalized proper nouns.
    """
    known_terms = {
        "dharma", "karma", "moksha", "mokṣa", "yoga", "bhakti", "arjuna",
        "krishna", "kṛṣṇa", "krsna", "dhritarashtra", "dhṛtarāṣṭra", "sanjaya",
        "sañjaya", "bhisma", "bhīṣma", "drona", "pandava", "pāṇḍava", "kaurava",
        "kaurava", "gandiva", "gāṇḍīva", "hanuman", "hanumān", "conch", "conchshell",
        "chariot", "bow", "flag", "grief", "compassion", "battle", "battlefield",
        "war", "fight", "duty", "action", "knowledge", "devotion", "renunciation",
        "sacrifice", "meditation", "self", "soul", "atman", "ātman", "brahman",
        "brahmana", "guna", "guṇa", "prakriti", "prakṛti", "purusha", "puruṣa",
        "maya", "māyā", "avidya", "avidyā", "vidya", "vidyā", "sattva", "rajas",
        "tamas", "tāmas", "rajas", "sattva", "visnu", "viṣṇu", "shiva", "śiva",
        "rudra", "indra", "varuna", "yama", "agni", "surya", "sūrya", "chandra",
        "candra", "vayu", "vāyu", "kubera", "ishvara", "īśvara", "guru",
        "disciple", "teacher", "student", "king", "warrior", "kshatriya",
        "kṣatriya", "brahmin", "brāhmaṇa", "vaishya", "vaiśya", "shudra",
        "śūdra", "caste", "varna", "ashrama", "āśrama", "stage", "life",
        "death", "rebirth", "reincarnation", "liberation", "salvation",
        "grace", "mercy", "wrath", "anger", "desire", "attachment", "detachment",
        "equanimity", "peace", "joy", "sorrow", "fear", "courage", "strength",
        "weakness", "ignorance", "wisdom", "truth", "falsehood", "righteousness",
        "unrighteousness", "sin", "virtue", "merit", "demerit", "fate", "destiny",
        "free will", "choice", "consequence", "reward", "punishment", "heaven",
        "hell", "earth", "world", "universe", "cosmos", "creation", "destruction",
        "preservation", "manifestation", "unmanifest", "eternal", "temporary",
        "permanent", "impermanent", "change", "unchanging", "form", "formless",
        "name", "form", "quality", "attribute", "property", "characteristic",
        "nature", "essence", "substance", "matter", "spirit", "consciousness",
        "mind", "intellect", "ego", "senses", "sense objects", "perception",
        "cognition", "memory", "imagination", "thought", "emotion", "feeling",
        "will", "intention", "purpose", "goal", "means", "end", "path", "way",
        "method", "practice", "discipline", "technique", "process", "procedure",
        "rule", "principle", "law", "order", "chaos", "harmony", "balance",
        "proportion", "measure", "limit", "boundary", "center", "periphery",
        "whole", "part", "unity", "diversity", "multiplicity", "singularity",
        "plurality", "duality", "non-duality", "advaita", "dvaita", "vishishtadvaita",
        "qualified", "absolute", "relative", "subjective", "objective", "personal",
        "impersonal", "transcendent", "immanent", "internal", "external", "inner",
        "outer", "higher", "lower", "superior", "inferior", "best", "worst",
        "good", "bad", "evil", "beautiful", "ugly", "pleasant", "unpleasant",
        "agreeable", "disagreeable", "favorable", "unfavorable", "auspicious",
        "inauspicious", "lucky", "unlucky", "fortunate", "unfortunate", "happy",
        "unhappy", "content", "discontent", "satisfied", "dissatisfied", "fulfilled",
        "unfulfilled", "complete", "incomplete", "perfect", "imperfect", "flawless",
        "flawed", "pure", "impure", "clean", "unclean", "holy", "unholy", "sacred",
        "profane", "divine", "mundane", "spiritual", "material", "physical",
        "mental", "emotional", "intellectual", "intuitive", "instinctive",
        "voluntary", "involuntary", "conscious", "unconscious", "subconscious",
        "aware", "unaware", "alert", "drowsy", "awake", "asleep", "dreaming",
        "deep sleep", "waking", "dreamless", "lucid", "vision", "dream", "nightmare",
        "apparition", "ghost", "spirit", "demon", "angel", "deity", "god", "goddess",
        "deva", "devi", "asura", "rakshasa", "rākṣasa", "yaksha", "yakṣa",
        "gandharva", "apsara", "naga", "nāga", "garuda", "serpent", "snake",
        "elephant", "horse", "lion", "tiger", "bull", "cow", "bird", "fish",
        "tree", "flower", "fruit", "seed", "root", "branch", "leaf", "stem",
        "mountain", "river", "ocean", "sea", "lake", "pond", "stream", "waterfall",
        "forest", "desert", "plain", "valley", "cave", "city", "village", "town",
        "palace", "temple", "ashram", "hermitage", "monastery", "school", "university",
        "library", "book", "text", "scripture", "veda", "vedas", "upanishad",
        "upanishads", "gita", "gītā", "bhagavad", "ramayana", "mahabharata",
        "mahābhārata", "purana", "purāṇa", "sutra", "sūtra", "shastra", "śāstra",
        "mantra", "yantra", "tantra", "stotra", "hymn", "prayer", "chant",
        "recitation", "reading", "study", "learning", "teaching", "instruction",
        "guidance", "advice", "counsel", "direction", "command", "order", "request",
        "question", "answer", "reply", "response", "statement", "declaration",
        "assertion", "claim", "argument", "debate", "discussion", "conversation",
        "dialogue", "discourse", "lecture", "sermon", "speech", "talk", "words",
        "speech", "language", "grammar", "syntax", "semantics", "phonetics",
        "phonology", "morphology", "etymology", "lexicon", "vocabulary", "dictionary",
        "thesaurus", "encyclopedia", "commentary", "explanation", "interpretation",
        "translation", "transliteration", "transcription", "notation", "symbol",
        "sign", "mark", "character", "letter", "syllable", "word", "phrase",
        "sentence", "paragraph", "chapter", "section", "part", "volume", "book",
        "work", "composition", "creation", "production", "generation", "origin",
        "source", "cause", "reason", "motive", "purpose", "intention", "design",
        "plan", "scheme", "project", "program", "system", "structure", "organization",
        "arrangement", "order", "sequence", "series", "chain", "link", "connection",
        "relation", "relationship", "association", "correlation", "correspondence",
        "analogy", "comparison", "contrast", "difference", "similarity", "likeness",
        "identity", "distinction", "separation", "division", "partition", "boundary",
        "limit", "edge", "border", "frontier", "threshold", "door", "gate",
        "entrance", "exit", "path", "road", "way", "route", "journey", "travel",
        "voyage", "expedition", "pilgrimage", "quest", "search", "seek", "find",
        "discover", "reveal", "uncover", "expose", "show", "display", "exhibit",
        "present", "offer", "give", "receive", "take", "accept", "reject",
        "refuse", "deny", "admit", "confess", "acknowledge", "recognize",
        "identify", "name", "call", "address", "speak", "say", "tell", "ask",
        "answer", "respond", "reply", "retort", "rejoin", "counter", "oppose",
        "resist", "fight", "struggle", "contend", "compete", "rival", "challenge",
        "defy", "confront", "face", "meet", "encounter", "experience", "undergo",
        "suffer", "endure", "bear", "tolerate", "accept", "submit", "yield",
        "surrender", "give up", "abandon", "forsake", "leave", "depart", "go",
        "come", "arrive", "reach", "attain", "achieve", "accomplish", "complete",
        "finish", "end", "terminate", "conclude", "close", "stop", "cease",
        "pause", "rest", "relax", "calm", "quiet", "still", "silent", "peaceful",
        "tranquil", "serene", "composed", "collected", "centered", "balanced",
        "harmonious", "unified", "integrated", "whole", "complete", "total",
        "entire", "full", "perfect", "ideal", "supreme", "highest", "greatest",
        "ultimate", "absolute", "infinite", "eternal", "everlasting", "permanent",
        "unchanging", "immutable", "constant", "stable", "fixed", "steady",
        "firm", "solid", "strong", "powerful", "mighty", "forceful", "energetic",
        "dynamic", "active", "vibrant", "lively", "animated", "spirited",
        "enthusiastic", "eager", "keen", "ardent", "passionate", "fervent",
        "zealous", "devoted", "dedicated", "committed", "loyal", "faithful",
        "true", "honest", "sincere", "genuine", "authentic", "real", "actual",
        "factual", "accurate", "correct", "right", "proper", "appropriate",
        "suitable", "fitting", "apt", "relevant", "pertinent", "applicable",
        "useful", "helpful", "beneficial", "advantageous", "profitable",
        "valuable", "precious", "priceless", "invaluable", "worthwhile",
        "meaningful", "significant", "important", "essential", "necessary",
        "vital", "crucial", "critical", "key", "fundamental", "basic",
        "primary", "principal", "main", "chief", "leading", "foremost",
        "preeminent", "paramount", "supreme", "sovereign", "independent",
        "autonomous", "self-sufficient", "self-reliant", "self-contained",
        "self-existent", "self-luminous", "self-aware", "self-conscious",
        "self-realized", "self-actualized", "self-fulfilled", "self-perfected",
        "enlightened", "awakened", "liberated", "free", "released", "delivered",
        "saved", "redeemed", "rescued", "protected", "guarded", "defended",
        "shielded", "sheltered", "covered", "hidden", "concealed", "secret",
        "mysterious", "wonderful", "marvelous", "miraculous", "amazing",
        "astonishing", "astounding", "surprising", "startling", "shocking",
        "stunning", "breathtaking", "awe-inspiring", "majestic", "grand",
        "glorious", "splendid", "magnificent", "sublime", "transcendent",
        "divine", "sacred", "holy", "blessed", "auspicious", "fortunate",
        "prosperous", "abundant", "plentiful", "rich", "wealthy", "opulent",
        "lavish", "generous", "liberal", "bountiful", "munificent", "charitable",
        "kind", "compassionate", "merciful", "gracious", "forgiving", "tolerant",
        "patient", "forbearing", "gentle", "mild", "soft", "tender", "loving",
        "caring", "nurturing", "supportive", "encouraging", "uplifting",
        "inspiring", "motivating", "stimulating", "invigorating", "refreshing",
        "reviving", "renewing", "restoring", "healing", "curing", "mending",
        "repairing", "fixing", "correcting", "rectifying", "adjusting",
        "modifying", "changing", "transforming", "converting", "transmuting",
        "evolving", "developing", "growing", "expanding", "increasing",
        "multiplying", "proliferating", "spreading", "extending", "stretching",
        "reaching", "touching", "contacting", "connecting", "joining",
        "uniting", "merging", "blending", "mixing", "combining", "integrating",
        "synthesizing", "harmonizing", "balancing", "equalizing", "leveling",
        "flattening", "smoothing", "polishing", "refining", "purifying",
        "cleansing", "washing", "bathing", "immersing", "dipping", "plunging",
        "diving", "sinking", "falling", "descending", "dropping", "lowering",
        "reducing", "diminishing", "decreasing", "lessening", "shrinking",
        "contracting", "compressing", "condensing", "concentrating", "focusing",
        "centering", "aiming", "directing", "pointing", "indicating", "showing",
        "demonstrating", "illustrating", "exemplifying", "representing",
        "symbolizing", "signifying", "meaning", "denoting", "connoting",
        "implying", "suggesting", "hinting", "intimating", "insinuating",
        "alluding", "referring", "mentioning", "citing", "quoting", "repeating",
        "echoing", "resonating", "vibrating", "sounding", "ringing", "tolling",
        "pealing", "chiming", "clanging", "clashing", "crashing", "thundering",
        "roaring", "bellowing", "shouting", "yelling", "screaming", "crying",
        "weeping", "sobbing", "wailing", "lamenting", "mourning", "grieving",
        "sorrowing", "regretting", "repenting", "atoning", "expiating",
        "purifying", "cleansing", "absolving", "forgiving", "pardoning",
        "excusing", "overlooking", "ignoring", "disregarding", "neglecting",
        "forgetting", "omitting", "missing", "losing", "dropping", "falling",
        "failing", "faltering", "stumbling", "tripping", "slipping", "sliding",
        "gliding", "floating", "flying", "soaring", "rising", "ascending",
        "climbing", "scaling", "mounting", "surmounting", "overcoming",
        "conquering", "defeating", "vanquishing", "subduing", "subjugating",
        "mastering", "controlling", "commanding", "directing", "guiding",
        "leading", "conducting", "managing", "administering", "governing",
        "ruling", "reigning", "dominating", "prevailing", "triumphing",
        "succeeding", "winning", "achieving", "attaining", "reaching",
        "arriving", "coming", "approaching", "nearing", "drawing close",
        "advancing", "progressing", "proceeding", "moving", "going",
        "traveling", "journeying", "wandering", "roaming", "rambling",
        "strolling", "walking", "stepping", "pacing", "marching", "striding",
        "running", "rushing", "hurrying", "speeding", "racing", "flying",
        "darting", "darting", "shooting", "propelling", "launching",
        "projecting", "throwing", "casting", "hurling", "flinging",
        "tossing", "pitching", "lobbing", "slinging", "swinging",
        "whirling", "spinning", "rotating", "revolving", "turning",
        "twisting", "winding", "coiling", "curling", "looping",
        "spiraling", "corkscrewing", "screw", "drill", "bore",
        "pierce", "penetrate", "enter", "invade", "infiltrate",
        "permeate", "pervade", "saturate", "soak", "drench",
        "douse", "wet", "moisten", "dampen", "humidify",
    }
    
    words = re.findall(r'\b\w+\b', query.lower())
    lemmas = []
    for word in words:
        if word in known_terms:
            lemmas.append(word)
    
    iast_words = re.findall(r'\b[a-zāīūṛṝḷḹṃḥṅñṭḍṇśṣ]+\b', query)
    for word in iast_words:
        if word not in lemmas:
            lemmas.append(word.lower())
    
    return lemmas[:20]


@dataclass
class HybridSearchResult:
    """Container for hybrid search results."""
    id: str
    text: str
    final_score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    colbert_score: float = 0.0
    bm25_score: float = 0.0
    exact_score: float = 0.0
    dataset_type: str = ""
    verse_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "final_score": self.final_score,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "colbert_score": self.colbert_score,
            "bm25_score": self.bm25_score,
            "exact_score": self.exact_score,
            "dataset_type": self.dataset_type,
            "verse_id": self.verse_id,
            "metadata": self.metadata
        }


@dataclass
class RerankerWeights:
    """Weights for different retrieval methods."""
    dense: float = 1.0
    sparse: float = 0.8
    colbert: float = 0.9
    bm25: float = 0.7
    
    def to_array(self) -> np.ndarray:
        return np.array([self.dense, self.sparse, self.colbert, self.bm25])
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "dense": self.dense,
            "sparse": self.sparse,
            "colbert": self.colbert,
            "bm25": self.bm25
        }

    def normalize(self) -> "RerankerWeights":
        total = self.dense + self.sparse + self.colbert + self.bm25
        if total > 0:
            return RerankerWeights(
                dense=self.dense / total,
                sparse=self.sparse / total,
                colbert=self.colbert / total,
                bm25=self.bm25 / total
            )
        return self


class HybridRetriever:
    """Hybrid retrieval with split paths:

    Neo4j (Graph DB): Verse-level + multi-hop graph traversal on morphosyntax data
    Qdrant (Vector DB): Semantic relevance via dense/sparse/BM25 on segmented+lemmatized data
    """

    def __init__(
        self,
        embedding_client: NVIDIAEmbeddingClient,
        qdrant_manager: QdrantManager = None,
        neo4j_manager: Neo4jManager = None,
        weights: Optional[RerankerWeights] = None,
        l1_lambda: float = L1_REG_LAMBDA,
        l2_lambda: float = L2_REG_LAMBDA,
        llm_client: Any = None,
    ):
        self.embedding_client = embedding_client
        self.qdrant_manager = qdrant_manager
        self.neo4j_manager = neo4j_manager
        self.weights = weights or RerankerWeights()
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.llm_client = llm_client
        self._neo4j_available = neo4j_manager is not None
        self._qdrant_available = qdrant_manager is not None
    
    def embed_query(self, query: str) -> EmbeddingResult:
        """Embed a query using all three embedding types."""
        return self.embedding_client.embed_query(query)
    
    # ============================================================
    # NEO4J — VERSE-LEVEL + MULTI-HOP GRAPH RETRIEVAL
    # ============================================================
    
    def _neo4j_verse_level(
        self,
        verse_filter: VerseFilter,
        top_k: int = 10,
    ) -> List[Neo4jSearchResult]:
        """Neo4j: Direct verse lookup or range-based retrieval."""
        if not self.neo4j_manager:
            return []
        
        if verse_filter.verse_ids and len(verse_filter.verse_ids) <= 5:
            results = []
            for vid in verse_filter.verse_ids:
                results.extend(self.neo4j_manager.search_by_verse_id(vid))
            return results[:top_k * 2]
        
        if verse_filter.chapter:
            vs = verse_filter.verse_start or 1
            ve = verse_filter.verse_end or vs
            return self.neo4j_manager.search_by_verse_range(
                verse_filter.chapter, vs, ve,
            )
        
        return []
    
    def _neo4j_multi_hop(
        self,
        query: str,
        verse_filter: VerseFilter,
        top_k: int = 10,
        query_embedding: Optional[EmbeddingResult] = None,
    ) -> List[Neo4jSearchResult]:
        """Neo4j: Multi-hop graph traversal via lemmas."""
        if not self.neo4j_manager:
            return []
        
        lemmas = extract_sanskrit_lemmas(query)
        
        if lemmas:
            results = self.neo4j_manager.search_multi_hop(lemmas, top_k=top_k * 2, max_hops=2)
            
            if verse_filter.has_filter():
                filtered = []
                verse_ids_set = set(verse_filter.verse_ids) if verse_filter.verse_ids else set()
                for r in results:
                    if not verse_ids_set or (r.verse_id and r.verse_id in verse_ids_set):
                        filtered.append(r)
                if filtered:
                    return filtered[:top_k]
            
            return results[:top_k]
        
        if verse_filter.has_filter():
            return self._neo4j_verse_level(verse_filter, top_k)
        
        query_embedding = query_embedding or self.embed_query(query)
        if not (query_embedding.metadata or {}).get("dense_available", True):
            return []

        return self.neo4j_manager.search_dense(
            query_embedding.dense_vector,
            top_k=top_k,
        )
    
    # ============================================================
    # QDRANT — SEMANTIC RELEVANCE SEARCH
    # ============================================================
    
    def _qdrant_semantic(
        self,
        query: str,
        query_embedding: EmbeddingResult,
        verse_filter: VerseFilter,
        top_k: int = 10,
        include_bm25: bool = True,
        collection_name: Optional[str] = None,
    ) -> Tuple[
        List[QdrantSearchResult],
        List[QdrantSearchResult],
        List[QdrantSearchResult],
        List[QdrantSearchResult],
    ]:
        """Qdrant: Dense + sparse + BM25 semantic search plus exact verse fallback."""
        if not self.qdrant_manager:
            return [], [], [], []
        
        vf_dict = verse_filter.to_qdrant_dict() if verse_filter.has_filter() else None
        
        dense_results = []
        if (query_embedding.metadata or {}).get("dense_available", True):
            dense_results = self.qdrant_manager.search_dense(
                query_embedding.dense_vector,
                top_k=top_k * 2,
                verse_filter=vf_dict,
                collection_name=collection_name,
            )
        
        sparse_results = self.qdrant_manager.search_sparse(
            query_embedding.sparse_vector,
            top_k=top_k * 2,
            verse_filter=vf_dict,
            collection_name=collection_name,
        )
        
        bm25_results = []
        if include_bm25:
            bm25_results = self.qdrant_manager.bm25_search(
                re.findall(r"\w+", query.lower()),
                top_k=top_k,
                verse_filter=vf_dict,
                collection_name=collection_name,
            )

        exact_results = []
        if verse_filter.verse_ids:
            if collection_name:
                exact_results = self.qdrant_manager.search_by_verse_ids(
                    verse_filter.verse_ids,
                    top_k=top_k * 2,
                    collection_name=collection_name,
                )
            else:
                exact_results = self.qdrant_manager.search_by_verse_ids(
                    verse_filter.verse_ids,
                    top_k=top_k * 2,
                )

        return dense_results, sparse_results, bm25_results, exact_results
    
    # ============================================================
    # RRF FUSION
    # ============================================================
    
    def rrf_fusion(
        self,
        result_lists: List[List[Any]],
        weights: np.ndarray = None,
        k: int = RRF_K,
        key_func: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Reciprocal Rank Fusion for combining multiple result lists."""
        if weights is None:
            weights = np.ones(len(result_lists))
        
        rrf_scores = defaultdict(float)
        
        for weight, results in zip(weights, result_lists):
            for rank, result in enumerate(results):
                key = key_func(result) if key_func else result.id
                rrf_scores[key] += weight / (k + rank + 1)
        
        return dict(sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True))
    
    def apply_l1_regularization(self, scores: np.ndarray) -> np.ndarray:
        """Apply L1 regularization to prevent overfitting to any single method."""
        weights = self.weights.to_array()
        l1_penalty = self.l1_lambda * np.sum(np.abs(weights))
        return scores - l1_penalty
    
    def apply_l2_regularization(self, scores: np.ndarray) -> np.ndarray:
        """Apply L2 regularization to smooth the score distribution."""
        weights = self.weights.to_array()
        l2_penalty = self.l2_lambda * np.sum(weights ** 2)
        return scores - l2_penalty
    
    def apply_combined_regularization(
        self,
        scores: np.ndarray,
        method_weights: np.ndarray
    ) -> np.ndarray:
        """Apply both L1 and L2 regularization."""
        weights = self.weights.to_array()
        total_penalty = (
            self.l1_lambda * np.sum(np.abs(weights)) +
            self.l2_lambda * np.sum(weights ** 2)
        )
        return scores * (1 - total_penalty)
    
    def normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return scores
        
        max_score = max(scores.values())
        min_score = min(scores.values())
        
        if max_score == min_score:
            return {k: 0.5 for k in scores}
        
        return {
            k: (v - min_score) / (max_score - min_score)
            for k, v in scores.items()
        }
    
    # ============================================================
    # MAIN RETRIEVAL — SPLIT PATHS
    # ============================================================

    def cross_db_rrf_search(
        self,
        query: str,
        top_k: int = 10,
        k_rrf: int = RRF_K,
        include_bm25: bool = True,
        regularization: str = "combined",
        verse_filter: Optional[VerseFilter] = None,
        collection_name: Optional[str] = None,
    ) -> List[HybridSearchResult]:
        """Retrieve from Neo4j (verse-level graph) + Qdrant (semantic vectors),
        then fuse with RRF and optional LLM re-ranking.
        """
        if verse_filter is None:
            verse_filter = parse_verse_references(query)

        query_embedding = self.embed_query(query)

        neo4j_results = []
        qdrant_dense = []
        qdrant_sparse = []
        qdrant_bm25 = []
        qdrant_exact = []

        # Neo4j: verse-level + multi-hop graph traversal (only if available)
        if self._neo4j_available and self.neo4j_manager:
            if verse_filter.has_filter():
                neo4j_results = self._neo4j_verse_level(verse_filter, top_k)
                if not neo4j_results:
                    neo4j_results = self._neo4j_multi_hop(
                        query,
                        verse_filter,
                        top_k,
                        query_embedding=query_embedding,
                    )
            else:
                neo4j_results = self._neo4j_multi_hop(
                    query,
                    verse_filter,
                    top_k,
                    query_embedding=query_embedding,
                )

        # Qdrant: semantic relevance (dense + sparse + BM25) (only if available)
        if self._qdrant_available and self.qdrant_manager:
            qdrant_dense, qdrant_sparse, qdrant_bm25, qdrant_exact = self._qdrant_semantic(
                query, query_embedding, verse_filter, top_k, include_bm25, collection_name=collection_name,
            )

        # Build score maps
        def to_score_map(results):
            return {r.id: r.score for r in results}

        all_score_maps = [
            to_score_map(neo4j_results),
            to_score_map(qdrant_dense),
            to_score_map(qdrant_sparse),
        ]
        if include_bm25:
            all_score_maps.append(to_score_map(qdrant_bm25))

        if regularization in ['l1', 'l2', 'combined']:
            weights_arr = self.weights.to_array()
            for scores in all_score_maps:
                if scores:
                    scores_array = np.array(list(scores.values()))
                    if regularization == 'l1':
                        scores_array = self.apply_l1_regularization(scores_array)
                    elif regularization == 'l2':
                        scores_array = self.apply_l2_regularization(scores_array)
                    else:
                        scores_array = self.apply_combined_regularization(scores_array, weights_arr)
                    for i, key in enumerate(scores.keys()):
                        scores[key] = float(max(0, scores_array[i]))

        # RRF weights: adaptive based on available databases and verse filter
        neo4j_w = 0.0
        qdrant_w = 1.0

        if self._neo4j_available and self._qdrant_available:
            if verse_filter.has_filter():
                neo4j_w = RRF_WEIGHTS_VERSE_FILTER["neo4j"]
                qdrant_w = RRF_WEIGHTS_VERSE_FILTER["qdrant"]
            else:
                neo4j_w = RRF_WEIGHTS_NO_FILTER["neo4j"]
                qdrant_w = RRF_WEIGHTS_NO_FILTER["qdrant"]
        elif self._neo4j_available:
            neo4j_w = 1.0
            qdrant_w = 0.0

        result_lists = []
        rrf_weights_list = []

        if self._neo4j_available and neo4j_results:
            result_lists.append(neo4j_results)
            rrf_weights_list.append(neo4j_w)

        if self._qdrant_available:
            if qdrant_dense:
                result_lists.append(qdrant_dense)
                rrf_weights_list.append(qdrant_w * 0.6)
            if qdrant_sparse:
                result_lists.append(qdrant_sparse)
                rrf_weights_list.append(qdrant_w * 0.4)
            if include_bm25 and qdrant_bm25:
                result_lists.append(qdrant_bm25)
                rrf_weights_list.append(qdrant_w * 0.3)
            if qdrant_exact:
                result_lists.append(qdrant_exact)
                rrf_weights_list.append(qdrant_w * (0.5 if verse_filter.has_filter() else 0.2))

        rrf_weights = np.array(rrf_weights_list) if rrf_weights_list else np.array([1.0])

        if not result_lists:
            return []

        def fusion_key(result):
            return result.verse_id or result.id

        fused_scores = self.rrf_fusion(result_lists, rrf_weights, k=k_rrf, key_func=fusion_key)
        
        # Build result map
        result_map = {}
        for results, score_key in [
            (neo4j_results, 'neo4j_score'),
            (qdrant_dense, 'dense_score'),
            (qdrant_sparse, 'sparse_score'),
            (qdrant_bm25, 'bm25_score'),
            (qdrant_exact, 'exact_score'),
        ]:
            for r in results:
                key = fusion_key(r)
                if key not in result_map:
                    result_map[key] = {
                        'id': r.id,
                        'text': r.text,
                        'dataset_type': r.dataset_type,
                        'verse_id': r.verse_id,
                        'metadata': r.metadata,
                        'representative_score': r.score,
                        'neo4j_score': 0.0,
                        'dense_score': 0.0,
                        'sparse_score': 0.0,
                        'bm25_score': 0.0,
                        'exact_score': 0.0,
                    }
                elif r.score > result_map[key].get('representative_score', 0.0):
                    result_map[key].update({
                        'id': r.id,
                        'text': r.text,
                        'dataset_type': r.dataset_type,
                        'verse_id': r.verse_id,
                        'metadata': {**result_map[key].get('metadata', {}), **(r.metadata or {})},
                        'representative_score': r.score,
                    })
                result_map[key][score_key] = max(result_map[key].get(score_key, 0.0), r.score)
        
        # Build final results
        hybrid_results = []
        min_rrf_score = 0.0 if verse_filter.has_filter() else MIN_RRF_SCORE
        for doc_id, rrf_score in fused_scores.items():
            if rrf_score < min_rrf_score:
                continue
            if doc_id in result_map:
                rd = result_map[doc_id]
                hybrid_results.append(HybridSearchResult(
                    id=doc_id,
                    text=rd['text'],
                    final_score=rrf_score,
                    dense_score=rd['dense_score'],
                    sparse_score=rd['sparse_score'],
                    bm25_score=rd['bm25_score'],
                    exact_score=rd['exact_score'],
                    dataset_type=rd['dataset_type'],
                    verse_id=rd['verse_id'],
                    metadata={
                        **rd['metadata'],
                        'rrf_score': rrf_score,
                        'fusion_key': doc_id,
                        'fusion_level': 'verse' if rd['verse_id'] else 'document',
                        'neo4j_score': rd['neo4j_score'],
                        'qdrant_modes': [
                            mode
                            for mode, contributed in [
                                ('dense', rd['dense_score'] > 0),
                                ('sparse', rd['sparse_score'] > 0),
                                ('bm25', rd['bm25_score'] > 0),
                                ('exact_verse', rd['exact_score'] > 0),
                            ]
                            if contributed
                        ],
                        'sources': {
                            'qdrant': (
                                rd['dense_score'] > 0
                                or rd['sparse_score'] > 0
                                or rd['bm25_score'] > 0
                                or rd['exact_score'] > 0
                            ),
                            'neo4j': rd['neo4j_score'] > 0,
                        },
                        'verse_filter_applied': verse_filter.has_filter(),
                    }
                ))
        
        # LLM re-ranking if available and we have enough candidates
        if self.llm_client and self.llm_client.is_available() and len(hybrid_results) > 3:
            hybrid_results = self._llm_rerank(query, hybrid_results, top_k)
        
        return hybrid_results[:top_k]
    
    def _llm_rerank(
        self,
        query: str,
        candidates: List[HybridSearchResult],
        top_k: int = 10,
    ) -> List[HybridSearchResult]:
        """Re-rank candidates using LLM relevance scoring."""
        if not self.llm_client or not candidates:
            return candidates
        
        rerank_candidates = candidates[:15]
        
        candidate_texts = ""
        for i, c in enumerate(rerank_candidates):
            text_preview = c.text[:200].replace("\n", " ")
            candidate_texts += f"[{i+1}] (ID: {c.id}, Verse: {c.verse_id})\n{text_preview}\n\n"
        
        prompt = f"""Score each candidate's relevance to the question on a scale of 1-5.

Question: {query}

Candidates:
{candidate_texts}

Respond with ONLY a JSON array of scores in order, e.g.: [5, 3, 4, 2, 1, 3, 4, 2, 1, 3, 2, 1, 1, 2, 1]"""
        
        try:
            response = self.llm_client._generate(prompt)
            scores_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if scores_match:
                llm_scores = json.loads(scores_match.group(0))
                if isinstance(llm_scores, list) and len(llm_scores) >= len(rerank_candidates):
                    for i, c in enumerate(rerank_candidates):
                        llm_score = min(5, max(1, llm_scores[i])) / 5.0
                        c.final_score = 0.4 * c.final_score + 0.6 * llm_score
                        c.metadata['llm_rerank_score'] = round(llm_scores[i], 1)
                        c.metadata['final_score_after_rerank'] = round(c.final_score, 4)
                    
                    rerank_candidates.sort(key=lambda x: x.final_score, reverse=True)
        except Exception as e:
            print(f"LLM re-ranking failed: {e}")
        
        return rerank_candidates[:top_k]
    
    # ============================================================
    # LEGACY METHODS (backward compatibility)
    # ============================================================
    
    def hybrid_search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        include_bm25: bool = True,
        regularization: str = "combined"
    ) -> List[HybridSearchResult]:
        """Legacy: delegates to cross_db_rrf_search."""
        return self.cross_db_rrf_search(
            query,
            top_k,
            include_bm25=include_bm25,
            regularization=regularization,
            collection_name=collection_name,
        )
    
    def search_all_collections(
        self,
        query: str,
        top_k: int = 10,
        include_bm25: bool = True,
        regularization: str = "combined"
    ) -> Dict[str, List[HybridSearchResult]]:
        """Search across all dataset collections."""
        results = {}
        for dtype, coll_name in COLLECTION_NAMES.items():
            results[dtype] = self.hybrid_search(
                query, coll_name, top_k, include_bm25, regularization
            )
        return results
    
    def get_retrieval_stats(
        self,
        results: List[HybridSearchResult]
    ) -> Dict[str, Any]:
        """Get statistics about retrieval results."""
        if not results:
            return {}
        
        stats = {
            "total_results": len(results),
            "avg_dense_score": float(np.mean([r.dense_score for r in results])),
            "avg_sparse_score": float(np.mean([r.sparse_score for r in results])),
            "avg_colbert_score": float(np.mean([r.colbert_score for r in results])),
            "avg_bm25_score": float(np.mean([r.bm25_score for r in results])),
            "avg_final_score": float(np.mean([r.final_score for r in results])),
        }
        
        return stats


class RegularizedRetriever(HybridRetriever):
    """Extended retriever with learnable regularization parameters."""
    
    def __init__(
        self,
        embedding_client: NVIDIAEmbeddingClient,
        qdrant_manager: QdrantManager = None,
        neo4j_manager: Neo4jManager = None,
        l1_lambda: float = 0.01,
        l2_lambda: float = 0.001,
        adaptive: bool = True,
        llm_client: Any = None,
    ):
        super().__init__(
            embedding_client, qdrant_manager, neo4j_manager,
            l1_lambda=l1_lambda, l2_lambda=l2_lambda,
            llm_client=llm_client,
        )
        self.adaptive = adaptive
        self._score_history: List[Dict[str, float]] = []
    
    def update_regularization(
        self,
        scores: Dict[str, float],
        performance_metric: float
    ):
        """Adaptively update regularization parameters based on performance."""
        if not self.adaptive:
            return
        
        self._score_history.append({
            'l1_lambda': self.l1_lambda,
            'l2_lambda': self.l2_lambda,
            'performance': performance_metric
        })
        
        if len(self._score_history) > 10:
            recent = self._score_history[-10:]
            performances = [h['performance'] for h in recent]
            
            if np.std(performances) < 0.05:
                self.l1_lambda *= 0.9
                self.l2_lambda *= 0.9
            else:
                self.l1_lambda *= 1.1
                self.l2_lambda *= 1.1
            
            self.l1_lambda = float(np.clip(self.l1_lambda, 0.0001, 0.1))
            self.l2_lambda = float(np.clip(self.l2_lambda, 0.00001, 0.01))
    
    def get_regularization_params(self) -> Dict[str, Any]:
        """Get current regularization parameters."""
        return {
            'l1_lambda': self.l1_lambda,
            'l2_lambda': self.l2_lambda,
            'weights': self.weights.to_dict() if hasattr(self.weights, 'to_dict') else None
        }


if __name__ == "__main__":
    from embedding_client import NVIDIAEmbeddingClient
    from qdrant_manager import QdrantManager
    from neo4j_manager import Neo4jManager
    
    emb_client = NVIDIAEmbeddingClient()
    qdrant_mgr = QdrantManager()
    neo4j_mgr = Neo4jManager()
    
    qdrant_mgr.connect()
    neo4j_mgr.connect()
    
    retriever = HybridRetriever(emb_client, qdrant_mgr, neo4j_mgr)
    
    query = "dharma kṣetra"
    results = retriever.cross_db_rrf_search(query, top_k=5)
    
    print(f"\nQuery: {query}")
    print(f"Results: {len(results)}")
    for r in results:
        print(f"\n  ID: {r.id}")
        print(f"  Text: {r.text[:80]}...")
        print(f"  Score: {r.final_score:.4f}")
        print(f"  D: {r.dense_score:.3f} | S: {r.sparse_score:.3f} | BM25: {r.bm25_score:.3f}")
    
    qdrant_mgr.disconnect()
    neo4j_mgr.disconnect()
