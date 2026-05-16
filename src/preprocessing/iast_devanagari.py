"""IAST to Devanagari transliteration using indic-transliteration library."""

from functools import lru_cache

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from src.utils.logger import logger


class IASTConverter:
    """Convert between IAST and Devanagari scripts for Sanskrit text."""

    def __init__(self):
        self._iast = sanscript.IAST
        self._devanagari = sanscript.DEVANAGARI
        logger.info("IASTConverter initialized with indic-transliteration")

    def iast_to_devanagari(self, text: str) -> str:
        """Convert IAST text to Devanagari.

        Args:
            text: Sanskrit text in IAST transliteration.

        Returns:
            Devanagari representation of the text.
        """
        if not text or not text.strip():
            return ""
        try:
            return transliterate(text, self._iast, self._devanagari)
        except Exception as e:
            logger.warning(f"Transliteration failed for: {text[:50]}... Error: {e}")
            return text

    def devanagari_to_iast(self, text: str) -> str:
        """Convert Devanagari text to IAST.

        Args:
            text: Sanskrit text in Devanagari script.

        Returns:
            IAST transliteration of the text.
        """
        if not text or not text.strip():
            return ""
        try:
            return transliterate(text, self._devanagari, self._iast)
        except Exception as e:
            logger.warning(f"Transliteration failed for: {text[:50]}... Error: {e}")
            return text

    def iast_to_devanagari_batch(self, texts: list[str]) -> list[str]:
        """Convert a batch of IAST texts to Devanagari.

        Args:
            texts: List of IAST texts.

        Returns:
            List of Devanagari texts.
        """
        return [self.iast_to_devanagari(t) for t in texts]

    def convert_verse_data(self, verse_data: dict) -> dict:
        """Add Devanagari fields to a verse data dictionary.

        Args:
            verse_data: Dictionary with IAST fields.

        Returns:
            Dictionary with added Devanagari fields.
        """
        result = verse_data.copy()

        if "text_iast" in result:
            result["text_devanagari"] = self.iast_to_devanagari(result["text_iast"])

        if "verse_lines_iast" in result:
            result["verse_lines_devanagari"] = self.iast_to_devanagari_batch(
                result["verse_lines_iast"]
            )

        if "commentary_iast" in result:
            result["commentary_devanagari"] = self.iast_to_devanagari(
                result["commentary_iast"]
            )

        return result


@lru_cache(maxsize=1)
def get_converter() -> IASTConverter:
    """Get singleton IASTConverter instance."""
    return IASTConverter()
