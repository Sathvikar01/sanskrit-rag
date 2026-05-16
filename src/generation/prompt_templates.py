"""Prompt templates for answer generation."""

SYSTEM_PROMPT = """You are a scholarly assistant specializing in the Bhagavad Gita. You provide accurate, well-cited answers based on the original Sanskrit text and traditional commentaries.

Your knowledge includes:
- The original 700 verses of the Bhagavad Gita across 18 chapters
- Commentaries by three renowned scholars:
  1. Sridhara Swamin (Gita-Subodhini) - Advaita tradition
  2. Visvanatha Chakravarti - Acintya-bhedabheda tradition
  3. Baladeva Vidyabhushana (Gita-Bhushana) - Acintya-bhedabheda tradition

Guidelines:
- Always cite specific verses using the format "BhG X.Y" (e.g., BhG 2.47)
- Use IAST transliteration for Sanskrit terms with their meanings
- Be precise and scholarly in your explanations

Answer structure:
1. First, give YOUR OWN synthesized explanation of the concept based on the retrieved verses. Explain what the verse teaches, its meaning, and its significance. Do NOT lead with commentaries.
2. Cite and quote the relevant verses within your explanation as support.
3. Only AFTER your main explanation, include a "Traditional Commentaries" section at the end listing relevant scholarly interpretations.
- Keep commentaries brief — summarize each in 1-2 sentences.
- If the user asks a factual question (who, what, when), answer it directly first before adding commentary context."""


VERSE_CONTEXT_TEMPLATE = """## Retrieved Verses

The following verses are relevant to the user's question, ordered by relevance. Use these as the primary basis for your answer. Focus on the most relevant verses first.

{verses_context}

---

Based on these verses, answer the user's question thoroughly. Cite the specific verses you reference.

IMPORTANT: Your main answer should be your own explanation synthesized from the verses. Do NOT simply quote commentaries as your answer. Only after completing your main explanation, add a brief "Traditional Commentaries" section at the end with 1-2 sentence summaries of each relevant scholar's interpretation."""


COMMENTARIES_CONTEXT_TEMPLATE = """## Traditional Commentaries

The following commentaries from traditional scholars are available for reference. Include only the most relevant ones as a brief section at the end of your answer, not as the main body.

{commentaries_context}"""


VERSE_ENTRY_TEMPLATE = """### {verse_ref} (Relevance: {confidence:.2f}){chunk_type_label}
**Sanskrit (IAST):**
{verse_text_iast}

**Devanagari:**
{verse_text_devanagari}
"""


COMMENTARY_ENTRY_TEMPLATE = """### {verse_ref} — {commentator_name} ({tradition}){chunk_type_label}
**Commentary:**
{commentary_text}
"""


def format_verse_context(reranked_results: list[dict]) -> str:
    """Format reranked results into separate verse and commentary sections.

    Verses form the primary answer context. Commentaries are provided
    as supplementary material at the end.

    Args:
        reranked_results: List of re-ranked results with verse text and commentaries.

    Returns:
        Tuple of (verses_context, commentaries_context).
    """
    commentator_info = {
        "sridhara": ("Sridhara Swamin", "Advaita"),
        "visvanatha": ("Visvanatha Chakravarti", "Acintya-bhedabheda"),
        "baladeva": ("Baladeva Vidyabhushana", "Acintya-bhedabheda"),
    }

    verse_chunks = [r for r in reranked_results if r.get("chunk_type") == "verse"]
    commentary_chunks = [r for r in reranked_results if r.get("chunk_type") == "commentary"]
    combined_chunks = [r for r in reranked_results if r.get("chunk_type") == "combined"]

    verse_entries = []
    commentary_entries = []
    seen_verses = set()
    seen_commentaries = set()

    # Verse chunks first
    for result in verse_chunks + combined_chunks:
        verse_ref = result.get("verse_ref", "Unknown")
        if verse_ref in seen_verses:
            continue
        seen_verses.add(verse_ref)

        chunk_type_label = f" [{result.get('chunk_type', '')}]" if result.get("chunk_type", "") != "verse" else ""
        entry = VERSE_ENTRY_TEMPLATE.format(
            verse_ref=verse_ref,
            confidence=result.get("confidence", {}).get("overall_confidence", 0.0),
            chunk_type_label=chunk_type_label,
            verse_text_iast=result.get("text_iast", ""),
            verse_text_devanagari=result.get("text_devanagari", ""),
        )
        verse_entries.append(entry)

    # Commentary chunks in a separate section
    for result in commentary_chunks:
        verse_ref = result.get("verse_ref", "Unknown")
        comm_key = result.get("commentator", "")
        comm_id = f"{verse_ref}:{comm_key}"
        if comm_id in seen_commentaries:
            continue
        seen_commentaries.add(comm_id)

        name, tradition = commentator_info.get(comm_key, (comm_key, ""))
        chunk_type_label = " [commentary]"
        entry = COMMENTARY_ENTRY_TEMPLATE.format(
            verse_ref=verse_ref,
            commentator_name=name,
            tradition=tradition,
            chunk_type_label=chunk_type_label,
            commentary_text=result.get("text_iast", ""),
        )
        commentary_entries.append(entry)

    # Also extract commentaries from verse/combined chunks that have inline commentary
    for result in verse_chunks + combined_chunks:
        verse_ref = result.get("verse_ref", "Unknown")
        comm_key = result.get("commentator", "")
        if not comm_key:
            continue
        comm_id = f"{verse_ref}:{comm_key}"
        if comm_id in seen_commentaries:
            continue
        seen_commentaries.add(comm_id)

        name, tradition = commentator_info.get(comm_key, (comm_key, ""))
        chunk_type_label = " [from verse chunk]"
        entry = COMMENTARY_ENTRY_TEMPLATE.format(
            verse_ref=verse_ref,
            commentator_name=name,
            tradition=tradition,
            chunk_type_label=chunk_type_label,
            commentary_text=result.get("text_iast", ""),
        )
        commentary_entries.append(entry)

    verses_context = "\n\n".join(verse_entries) if verse_entries else "No relevant verses found."
    commentaries_context = "\n\n".join(commentary_entries) if commentary_entries else ""

    return verses_context, commentaries_context


def build_generation_prompt(
    query: str,
    reranked_results: list[dict],
    include_devanagari: bool = True,
) -> str:
    """Build the complete prompt for answer generation.

    Args:
        query: User's original question.
        reranked_results: Re-ranked retrieval results.
        include_devanagari: Whether to include Devanagari text.

    Returns:
        Complete prompt string.
    """
    verses_context, commentaries_context = format_verse_context(reranked_results)

    user_prompt = f"""## User Question

{query}

{VERSE_CONTEXT_TEMPLATE.format(verses_context=verses_context)}"""

    if commentaries_context:
        user_prompt += f"\n\n{COMMENTARIES_CONTEXT_TEMPLATE.format(commentaries_context=commentaries_context)}"

    user_prompt += "\n\nPlease provide your answer. Start with your own explanation, then include Traditional Commentaries at the end."

    return user_prompt
