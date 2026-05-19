"""Prompt templates for answer generation."""

SYSTEM_PROMPT = """You are a Bhagavad Gita subject-matter expert. You must answer ONLY using the retrieved verses provided below. You have NO other knowledge source.

ABSOLUTE RULES:

1. **ONLY retrieved content**: Every claim, explanation, and reference in your answer MUST come from the retrieved verses provided. Do NOT use your own knowledge, training data, or general information. If the retrieved verses do not contain enough information to answer, say so explicitly.

2. **Explanation-first**: Start with a clear explanation of the concept. Write as if you are teaching — define terms, explain significance, give context.

3. **Every verse MUST be cited**: When you reference ANY information from a verse, you MUST include the verse reference (e.g., BhG 2.47) in your answer. No exceptions. Format: "**[BhG X.Y]**" right after the relevant explanation.

4. **Quote verses with full context**: When you cite a verse, quote the full Sanskrit (IAST) text provided to you, followed by the English translation provided. Then explain what it means in plain language.

5. **NO TABLES**: Do NOT use markdown tables. Use paragraphs, bullet points, and headings only.

6. **Commentary**: Do NOT discuss commentators in your main answer body. If traditional commentary is provided, add ONE brief section at the very end labeled "## Scholarly Context" with at most 2-3 sentences summarizing the single most relevant scholarly interpretation.

7. **Be direct**: Answer the question fully. Do not hedge or say "this is a complex topic" without explaining it.

8. **Sanskrit terms**: When you introduce a Sanskrit term, give the IAST transliteration followed by the Devanagari in parentheses and its English meaning. Example: dharma (धर्म — duty, righteousness).

9. **NO VERSES = NO ANSWER**: If the retrieved verses section says "No relevant verses found" or contains no actual verse text, you MUST respond with: "I don't have relevant verses from the Bhagavad Gita to answer this question. The retrieval system did not find supporting verses for this query." Do NOT answer from your own knowledge when no verses are provided.

10. **Answer structure**: Use this format:
   - Main explanation with inline verse citations
   - Key verse quotations (full Sanskrit + English translation) with explanations
   - Optional: "## Scholarly Context" section at the very end (only if commentary is provided)"""


VERSE_CONTEXT_TEMPLATE = """## Retrieved Verses (Primary Source Material)

These are the ONLY verses you may reference. Each verse includes its full Sanskrit text, Devanagari script, and English translation. You MUST cite the verse reference (BhG X.Y) whenever you use information from a verse.

{verses_context}"""


COMMENTARIES_CONTEXT_TEMPLATE = """## Traditional Commentary (Reference Only)

This is supplementary context. Do NOT discuss it in your main answer. Only reference it in a brief "Scholarly Context" section at the very end if it adds meaningful insight.

{commentaries_context}"""


VERSE_ENTRY_TEMPLATE = """### {verse_ref}
**Sanskrit (IAST):** {verse_text_iast}
**Devanagari:** {verse_text_devanagari}
**English Translation:** {verse_text_english}
"""


COMMENTARY_ENTRY_TEMPLATE = """### {verse_ref} — {commentator_name} ({tradition})
{commentary_text}
"""


def format_verse_context(reranked_results: list[dict]) -> tuple[str, str]:
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
    seen_verses = set()

    # Verse chunks first — these are the primary content
    for result in verse_chunks + combined_chunks:
        verse_ref = result.get("verse_ref", "Unknown")
        if verse_ref in seen_verses:
            continue
        seen_verses.add(verse_ref)

        entry = VERSE_ENTRY_TEMPLATE.format(
            verse_ref=verse_ref,
            verse_text_iast=result.get("text_iast", ""),
            verse_text_devanagari=result.get("text_devanagari", ""),
            verse_text_english=result.get("text_english", "Translation not available."),
        )
        verse_entries.append(entry)

    # Commentary — pick ONLY the single most relevant one
    best_commentary = None
    best_score = -1.0

    for result in commentary_chunks + combined_chunks:
        if not result.get("commentator"):
            continue
        score = result.get("confidence", {}).get("overall_confidence", 0.0)
        if score > best_score:
            best_score = score
            best_commentary = result

    commentaries_context = ""
    if best_commentary:
        comm_key = best_commentary.get("commentator", "")
        name, tradition = commentator_info.get(comm_key, (comm_key, ""))
        entry = COMMENTARY_ENTRY_TEMPLATE.format(
            verse_ref=best_commentary.get("verse_ref", "Unknown"),
            commentator_name=name,
            tradition=tradition,
            commentary_text=best_commentary.get("text_iast", ""),
        )
        commentaries_context = entry

    verses_context = "\n\n".join(verse_entries) if verse_entries else "No relevant verses found."

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

    return user_prompt
