"""Intent-specific answer templates for grounded LLM prompting."""
from __future__ import annotations

from typing import Dict


ANSWER_TEMPLATES: Dict[str, str] = {
    "explicit_verse_lookup": """Answer structure:
- Start with the requested verse ID(s).
- Quote or summarize only the supplied canonical verse text.
- Add a short commentary note only if commentary evidence is supplied.
- Cite verse IDs and commentary authors.""",
    "commentary_question": """Answer structure:
- State what the supplied commentary says.
- Distinguish commentary interpretation from canonical verse text.
- Cite commentary as "Author on Verse ID".
- If commentary is missing, say the provided evidence does not include commentary.""",
    "comparison_question": """Answer structure:
- Compare the requested ideas or persons side by side.
- Anchor every comparison point in a verse ID or commentary author.
- Avoid claims not visible in the supplied evidence.""",
    "character_entity_question": """Answer structure:
- Identify the person/name/title only from supplied verses/commentary.
- Explain the title or relationship briefly.
- Cite the supporting verse ID or commentary author.""",
    "source_inspection": """Answer structure:
- List the most relevant source verses first.
- Include why each source was selected.
- Do not synthesize beyond source inspection unless the evidence directly supports it.""",
    "summary_explanation": """Answer structure:
- Give a concise explanation of the teaching.
- Include the most relevant Sanskrit line or phrase if present.
- Add commentary only as supporting interpretation.
- Cite verse IDs throughout.""",
    "theme_concept_question": """Answer structure:
- Define the concept only from supplied evidence.
- Synthesize across the top canonical verses.
- Mention limitations if the evidence is narrow.
- Cite verse IDs and commentary authors.""",
}


def get_answer_template(intent: str) -> str:
    """Return the template for an intent, falling back to concept explanation."""
    return ANSWER_TEMPLATES.get(intent, ANSWER_TEMPLATES["theme_concept_question"])
