"""Prompt templates for answer generation."""

SYSTEM_PROMPT = """You are a scholarly assistant specializing in the Bhagavad Gita, one of the most important Hindu scriptures. You provide accurate, well-cited answers based on the original Sanskrit text and traditional commentaries.

Your knowledge includes:
- The original 700 verses of the Bhagavad Gita across 18 chapters
- Commentaries by three renowned scholars:
  1. Sridhara Swamin (Gita-Subodhini) - Advaita tradition
  2. Visvanatha Chakravarti - Acintya-bhedabheda tradition
  3. Baladeva Vidyabhushana (Gita-Bhushana) - Acintya-bhedabheda tradition

Guidelines:
- Always cite specific verses using the format "BhG X.Y" (e.g., BhG 2.47)
- When asked to define or explain a concept, START with the most authoritative verse that provides a clear definition
- Present definitions before expanding into commentary analysis
- If multiple commentaries offer different perspectives, present them fairly
- Use IAST transliteration for Sanskrit terms with their meanings
- Be precise and scholarly in your explanations
- Prioritize verse chunks (which contain definitions) over commentary chunks (which contain elaboration)
- When the user asks "what is X", give the definition first, then elaborate with commentaries"""

VERSE_CONTEXT_TEMPLATE = """## Retrieved Verses and Commentaries

The following verses and commentaries are relevant to the user's question. Verses are ordered by relevance. Focus on the most relevant verses first when constructing your answer.

{verses_context}

---

Based on these sources, answer the user's question thoroughly and cite specific verses.
If the question asks for a definition, start with the clearest definitional verse before discussing commentaries."""

VERSE_ENTRY_TEMPLATE = """### {verse_ref} (Relevance: {confidence:.2f}){chunk_type_label}
**Sanskrit (IAST):**
{verse_text_iast}

**Devanagari:**
{verse_text_devanagari}

{commentaries}"""

COMMENTARY_TEMPLATE = """**{commentator_name}'s Commentary ({tradition}):**
{commentary_text}

"""


def format_verse_context(reranked_results: list[dict]) -> str:
    """Format reranked results into context for the generator.

    Prioritizes verse chunks over commentary chunks for better definitions.

    Args:
        reranked_results: List of re-ranked results with verse text and commentaries.

    Returns:
        Formatted context string.
    """
    commentator_info = {
        "sridhara": ("Sridhara Swamin", "Advaita"),
        "visvanatha": ("Visvanatha Chakravarti", "Acintya-bhedabheda"),
        "baladeva": ("Baladeva Vidyabhushana", "Acintya-bhedabheda"),
    }

    verse_chunks = [r for r in reranked_results if r.get("chunk_type") == "verse"]
    commentary_chunks = [r for r in reranked_results if r.get("chunk_type") == "commentary"]
    combined_chunks = [r for r in reranked_results if r.get("chunk_type") == "combined"]

    ordered = verse_chunks + combined_chunks + commentary_chunks

    entries = []
    seen_verses = set()
    for result in ordered:
        verse_ref = result.get("verse_ref", "Unknown")
        chunk_type = result.get("chunk_type", "")
        chunk_type_label = f" [{chunk_type}]" if chunk_type != "verse" else ""

        if chunk_type == "commentary" and verse_ref in seen_verses:
            continue

        seen_verses.add(verse_ref)

        commentaries = ""
        if result.get("commentator"):
            comm_key = result["commentator"]
            name, tradition = commentator_info.get(comm_key, (comm_key, ""))
            commentaries = COMMENTARY_TEMPLATE.format(
                commentator_name=name,
                tradition=tradition,
                commentary_text=result.get("text_iast", ""),
            )

        entry = VERSE_ENTRY_TEMPLATE.format(
            verse_ref=verse_ref,
            confidence=result.get("confidence", {}).get("overall_confidence", 0.0),
            chunk_type_label=chunk_type_label,
            verse_text_iast=result.get("text_iast", ""),
            verse_text_devanagari=result.get("text_devanagari", ""),
            commentaries=commentaries,
        )
        entries.append(entry)

    return "\n\n".join(entries)


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
    verses_context = format_verse_context(reranked_results)

    user_prompt = f"""## User Question

{query}

{VERSE_CONTEXT_TEMPLATE.format(verses_context=verses_context)}

Please provide a comprehensive answer with verse citations."""

    return user_prompt
