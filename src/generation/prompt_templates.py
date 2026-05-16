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
- When discussing philosophical concepts, reference relevant verses
- If multiple commentaries offer different perspectives, present them fairly
- Use IAST transliteration for Sanskrit terms with their meanings
- Be precise and scholarly in your explanations"""

VERSE_CONTEXT_TEMPLATE = """## Retrieved Verses and Commentaries

The following verses and commentaries are relevant to the user's question:

{verses_context}

---

Based on these sources, answer the user's question thoroughly and cite specific verses."""

VERSE_ENTRY_TEMPLATE = """### {verse_ref} (Confidence: {confidence:.2f})
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

    entries = []
    for result in reranked_results:
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
            verse_ref=result.get("verse_ref", "Unknown"),
            confidence=result.get("confidence", {}).get("overall_confidence", 0.0),
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
