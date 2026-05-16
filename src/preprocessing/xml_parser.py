"""XML parser for Bhagavad Gita TEI-XML datasets."""

import re
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

from src.utils.logger import logger

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

COMMENTATOR_MARKERS = {
    "sridhara": [
        "Śrīdharaḥ -", "Śrīdharaḥ-", "Sridharah -",
        "Śrīdhara:", "Sridhara:", "śrīdhara:", "śrīdharaḥ:",
        "Śrīdharaḥ --", "Śrīdharaḥ--",
        "śrīdharaḥ :", "śrīdhara :", "Śrīdharaḥ :", "Śrīdhara :",
    ],
    "visvanatha": [
        "Viśvanāthaḥ -", "Viśvanāthaḥ-", "Visvanathah -",
        "Viśvanātha:", "Visvanatha:", "viśvanātha:", "viśvanāthaḥ:",
        "Viśvanāthaḥ --", "Viśvanāthaḥ--",
        "viśvanāthaḥ :", "viśvanātha :", "Viśvanāthaḥ :", "Viśvanātha :",
    ],
    "baladeva": [
        "Baladevaḥ --", "Baladevaḥ--", "Baladevah --",
        "Baladeva:", "baladeva:", "baladevaḥ:",
        "Baladevaḥ -", "Baladevaḥ-",
        "baladevaḥ :", "baladeva :", "Baladevaḥ :", "Baladeva :",
    ],
}


@dataclass
class VerseData:
    """Parsed data for a single verse."""
    ref: str  # e.g., "BhG 1.1"
    chapter_num: int
    verse_num: int
    verse_lines_iast: list[str] = field(default_factory=list)
    sridhara_lines: list[str] = field(default_factory=list)
    visvanatha_lines: list[str] = field(default_factory=list)
    baladeva_lines: list[str] = field(default_factory=list)
    speaker: str = ""


@dataclass
class MorphoData:
    """Morphological annotation data for a verse."""
    ref: str
    verse_lines: list[str] = field(default_factory=list)
    sridhara_lines: list[str] = field(default_factory=list)
    visvanatha_lines: list[str] = field(default_factory=list)
    baladeva_lines: list[str] = field(default_factory=list)


@dataclass
class SegmentationData:
    """Segmentation/lemma annotation data for a verse."""
    ref: str
    verse_lines: list[str] = field(default_factory=list)
    sridhara_lines: list[str] = field(default_factory=list)
    visvanatha_lines: list[str] = field(default_factory=list)
    baladeva_lines: list[str] = field(default_factory=list)


def parse_verse_ref(ref: str) -> tuple[int, int]:
    """Parse 'BhG 1.1' into (chapter_num, verse_num)."""
    match = re.match(r"BhG\s+(\d+)\.(\d+)", ref)
    if not match:
        raise ValueError(f"Invalid verse reference: {ref}")
    return int(match.group(1)), int(match.group(2))


def extract_text_from_element(element) -> str:
    """Extract all text content from an XML element, handling nested tags."""
    parts = []
    for node in element.iter():
        if node.text:
            parts.append(node.text)
        if node.tail:
            parts.append(node.tail)
    return " ".join(parts).strip()


def detect_speaker(verse_lines: list[str]) -> str:
    """Detect who is speaking based on verse content."""
    full_text = " ".join(verse_lines).lower()
    if "dhṛtarāṣṭra uvāca" in full_text or "dhṛtarāṣṭra uvāca" in full_text:
        return "dhṛtarāṣṭra"
    if "arjuna uvāca" in full_text or "arjuna uvāca" in full_text:
        return "arjuna"
    if "bhagavān uvāca" in full_text or "śrī bhagavān uvāca" in full_text:
        return "krishna"
    if "sañjaya uvāca" in full_text or "saṃjaya uvāca" in full_text:
        return "sanjaya"
    return ""


class XMLParser:
    """Parser for Bhagavad Gita TEI-XML files."""

    def __init__(
        self,
        main_xml: str | Path,
        morpho_xml: str | Path,
        segmentation_xml: str | Path,
    ):
        self.main_xml = Path(main_xml)
        self.morpho_xml = Path(morpho_xml)
        self.segmentation_xml = Path(segmentation_xml)

    def _parse_xml(self, filepath: Path) -> etree._Element:
        """Parse XML file and return root element."""
        logger.info(f"Parsing XML: {filepath}")
        tree = etree.parse(str(filepath))
        return tree.getroot()

    def _extract_div_content(self, div_element) -> dict:
        """Extract content from a single <div> element."""
        result = {
            "ref": None,
            "verse_lines": [],
            "commentaries": {"sridhara": [], "visvanatha": [], "baladeva": []},
        }

        current_section = "verse"
        current_commentator = None

        for elem in div_element:
            tag = elem.tag.replace(f"{{{TEI_NS['tei']}}}", "")

            if tag == "p":
                text = extract_text_from_element(elem)

                if not text:
                    continue

                if re.match(r"BhG\s+\d+\.\d+", text):
                    result["ref"] = text.strip()
                    continue

                is_commentator = False
                for comm_key, markers in COMMENTATOR_MARKERS.items():
                    for marker in markers:
                        if marker in text:
                            current_commentator = comm_key
                            current_section = "commentary"
                            is_commentator = True

                            commentary_after_marker = text.split(marker, 1)[-1].strip()
                            if commentary_after_marker:
                                result["commentaries"][comm_key].append(commentary_after_marker)
                            break
                    if is_commentator:
                        break

                if not is_commentator and current_commentator:
                    result["commentaries"][current_commentator].append(text)
                elif not is_commentator and current_section == "verse":
                    pass

            elif tag == "lg":
                lines = []
                for l_elem in elem.findall(f"{{{TEI_NS['tei']}}}l"):
                    line_text = extract_text_from_element(l_elem)
                    if line_text:
                        lines.append(line_text)

                if current_section == "verse" or current_commentator is None:
                    result["verse_lines"].extend(lines)
                elif current_commentator:
                    result["commentaries"][current_commentator].extend(lines)

        return result

    def _extract_morpho_div(self, div_element) -> dict:
        """Extract morphological annotations from a <div> element."""
        result = {
            "ref": None,
            "verse_lines": [],
            "commentaries": {"sridhara": [], "visvanatha": [], "baladeva": []},
        }

        current_commentator = None

        for elem in div_element:
            tag = elem.tag.replace(f"{{{TEI_NS['tei']}}}", "")

            if tag == "p":
                text = extract_text_from_element(elem)
                if not text:
                    continue

                if re.match(r"BhG\s+\d+\.\d+", text):
                    result["ref"] = text.strip()
                    continue

                is_commentator = False
                for comm_key, markers in COMMENTATOR_MARKERS.items():
                    marker_clean = markers[0].replace(" -", "").replace("--", "")
                    if marker_clean in text or any(m in text for m in markers):
                        current_commentator = comm_key
                        is_commentator = True
                        break

                if not is_commentator and current_commentator:
                    result["commentaries"][current_commentator].append(text)
                elif not is_commentator and text.strip():
                    if any(c in text for c in ["Case=", "Gender=", "Tense=", "Mood="]):
                        result["verse_lines"].append(text)

            elif tag == "lg":
                lines = []
                for l_elem in elem.findall(f"{{{TEI_NS['tei']}}}l"):
                    line_text = extract_text_from_element(l_elem)
                    if line_text:
                        lines.append(line_text)

                if current_commentator:
                    result["commentaries"][current_commentator].extend(lines)
                else:
                    result["verse_lines"].extend(lines)

        return result

    def _extract_segmentation_div(self, div_element) -> dict:
        """Extract segmentation/lemma data from a <div> element."""
        result = {
            "ref": None,
            "verse_lines": [],
            "commentaries": {"sridhara": [], "visvanatha": [], "baladeva": []},
        }

        current_commentator = None

        for elem in div_element:
            tag = elem.tag.replace(f"{{{TEI_NS['tei']}}}", "")

            if tag == "p":
                text = extract_text_from_element(elem)
                if not text:
                    continue

                if re.match(r"BhG\s+\d+\.\d+", text):
                    result["ref"] = text.strip()
                    continue

                is_commentator = False
                for comm_key, markers in COMMENTATOR_MARKERS.items():
                    marker_clean = markers[0].replace(" -", "").replace("--", "")
                    if marker_clean in text or any(m in text for m in markers):
                        current_commentator = comm_key
                        is_commentator = True
                        break

                if not is_commentator and current_commentator:
                    result["commentaries"][current_commentator].append(text)

            elif tag == "lg":
                lines = []
                for l_elem in elem.findall(f"{{{TEI_NS['tei']}}}l"):
                    line_text = extract_text_from_element(l_elem)
                    if line_text:
                        lines.append(line_text)

                if current_commentator:
                    result["commentaries"][current_commentator].extend(lines)
                else:
                    result["verse_lines"].extend(lines)

        return result

    def parse_main(self) -> list[VerseData]:
        """Parse the main dataset.xml file."""
        root = self._parse_xml(self.main_xml)
        body = root.find(".//tei:body", TEI_NS)
        if body is None:
            raise ValueError("Could not find <body> in main XML")

        verses = []
        for div in body.xpath(".//tei:div", namespaces=TEI_NS):
            if div is None:
                continue
            content = self._extract_div_content(div)
            if not content["ref"]:
                continue

            chapter_num, verse_num = parse_verse_ref(content["ref"])
            speaker = detect_speaker(content["verse_lines"])

            verse = VerseData(
                ref=content["ref"],
                chapter_num=chapter_num,
                verse_num=verse_num,
                verse_lines_iast=content["verse_lines"],
                sridhara_lines=content["commentaries"]["sridhara"],
                visvanatha_lines=content["commentaries"]["visvanatha"],
                baladeva_lines=content["commentaries"]["baladeva"],
                speaker=speaker,
            )
            verses.append(verse)

        logger.info(f"Parsed {len(verses)} verses from main XML")
        return verses

    def parse_morpho(self) -> list[MorphoData]:
        """Parse the morphosyntactic annotation XML."""
        root = self._parse_xml(self.morpho_xml)
        body = root.find(".//tei:body", TEI_NS)
        if body is None:
            raise ValueError("Could not find <body> in morpho XML")

        morpho_data = []
        for div in body.xpath(".//tei:div", namespaces=TEI_NS):
            if div is None:
                continue
            content = self._extract_morpho_div(div)
            if not content["ref"]:
                continue

            data = MorphoData(
                ref=content["ref"],
                verse_lines=content["verse_lines"],
                sridhara_lines=content["commentaries"]["sridhara"],
                visvanatha_lines=content["commentaries"]["visvanatha"],
                baladeva_lines=content["commentaries"]["baladeva"],
            )
            morpho_data.append(data)

        logger.info(f"Parsed {len(morpho_data)} entries from morpho XML")
        return morpho_data

    def parse_segmentation(self) -> list[SegmentationData]:
        """Parse the segmentation/lemma XML."""
        root = self._parse_xml(self.segmentation_xml)
        body = root.find(".//tei:body", TEI_NS)
        if body is None:
            raise ValueError("Could not find <body> in segmentation XML")

        seg_data = []
        for div in body.xpath(".//tei:div", namespaces=TEI_NS):
            if div is None:
                continue
            content = self._extract_segmentation_div(div)
            if not content["ref"]:
                continue

            data = SegmentationData(
                ref=content["ref"],
                verse_lines=content["verse_lines"],
                sridhara_lines=content["commentaries"]["sridhara"],
                visvanatha_lines=content["commentaries"]["visvanatha"],
                baladeva_lines=content["commentaries"]["baladeva"],
            )
            seg_data.append(data)

        logger.info(f"Parsed {len(seg_data)} entries from segmentation XML")
        return seg_data

    def parse_all(self) -> tuple[list[VerseData], list[MorphoData], list[SegmentationData]]:
        """Parse all three XML files and return aligned data."""
        verses = self.parse_main()
        morpho = self.parse_morpho()
        segmentation = self.parse_segmentation()
        return verses, morpho, segmentation
