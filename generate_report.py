"""Generate comprehensive SRAG project report as PDF."""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import os
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor, black, white, Color
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, ListFlowable, ListItem
)
from reportlab.platypus.flowables import HRFlowable

OUTPUT_DIR = Path("docs/report_assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = Path("docs/SRAG_Project_Report.pdf")

# Colors
SAFFRON = HexColor("#FF9933")
DEEP_SAFFRON = HexColor("#CC7A00")
GOLD = HexColor("#DAA520")
DARK_BG = HexColor("#1a1a2e")
LIGHT_BG = HexColor("#f5f5f5")
BLUE = HexColor("#2196F3")
GREEN = HexColor("#4CAF50")
RED = HexColor("#f44336")
PURPLE = HexColor("#9C27B0")
GRAY = HexColor("#666666")


def create_architecture_diagram():
    """Create the main pipeline architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title
    ax.text(5, 6.7, "SRAG Pipeline Architecture", fontsize=16, fontweight="bold",
            ha="center", va="center", color="#1a1a2e")

    # Boxes
    boxes = [
        (1.0, 5.5, 2.0, 0.8, "User Query\n(Any Language)", "#E3F2FD", "#1565C0"),
        (4.0, 5.5, 2.0, 0.8, "Query Processor\n(MiMo v2.5)", "#E8F5E9", "#2E7D32"),
        (7.0, 5.5, 2.0, 0.8, "Query Type\nClassifier", "#FFF3E0", "#E65100"),
        (1.0, 3.8, 2.0, 0.8, "FAISS Vector\nSearch", "#F3E5F5", "#7B1FA2"),
        (4.0, 3.8, 2.0, 0.8, "Neo4j Graph\nSearch", "#FFEBEE", "#C62828"),
        (7.0, 3.8, 2.0, 0.8, "BM25 Lexical\nSearch", "#E0F7FA", "#00838F"),
        (4.0, 2.5, 2.0, 0.8, "Hybrid Fusion\n(Adaptive RRF)", "#FFF8E1", "#F57F17"),
        (4.0, 1.2, 2.0, 0.8, "Linguistic\nRe-ranking (9F)", "#FCE4EC", "#AD1457"),
        (7.0, 1.2, 2.0, 0.8, "MiMo v2.5\nGeneration", "#E8EAF6", "#283593"),
        (1.0, 1.2, 2.0, 0.8, "LangGraph\nLoop Controller", "#E0F2F1", "#00695C"),
    ]

    for x, y, w, h, text, fc, ec in boxes:
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.05", facecolor=fc,
                              edgecolor=ec, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, fontsize=8, ha="center", va="center",
                fontweight="bold", color="#1a1a2e")

    # Arrows
    arrows = [
        (2.0, 5.5, 4.0, 5.5),   # Query -> Processor
        (5.0, 5.5, 7.0, 5.5),   # Processor -> Classifier
        (7.0, 5.1, 1.0, 4.2),   # Classifier -> FAISS
        (7.0, 5.1, 4.0, 4.2),   # Classifier -> Neo4j
        (7.0, 5.1, 7.0, 4.2),   # Classifier -> BM25
        (1.0, 3.4, 4.0, 2.9),   # FAISS -> Fusion
        (4.0, 3.4, 4.0, 2.9),   # Neo4j -> Fusion
        (7.0, 3.4, 4.0, 2.9),   # BM25 -> Fusion
        (4.0, 2.1, 4.0, 1.6),   # Fusion -> Rerank
        (5.0, 1.2, 7.0, 1.2),   # Rerank -> Generate
        (3.0, 1.2, 1.0, 1.2),   # Rerank -> LangGraph
        (1.0, 1.6, 1.0, 5.1),   # LangGraph -> Query (loop)
    ]

    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#455A64", lw=1.5))

    # Legend
    ax.text(5, 0.3, "Adaptive weights shift based on query type: concept_short | factual_short | complex_long | concept_medium | general_medium",
            fontsize=7, ha="center", va="center", style="italic", color="#666")

    plt.tight_layout()
    path = OUTPUT_DIR / "architecture.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def create_retrieval_weights_chart():
    """Create adaptive retrieval weights visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: Retrieval weights
    ax = axes[0]
    query_types = ["concept_short", "factual_short", "complex_long", "concept_medium", "general_medium"]
    vector_w = [0.50, 0.35, 0.40, 0.45, 0.45]
    graph_w = [0.30, 0.40, 0.20, 0.30, 0.25]
    bm25_w = [0.20, 0.25, 0.40, 0.25, 0.30]

    x = np.arange(len(query_types))
    width = 0.25

    ax.bar(x - width, vector_w, width, label="Vector", color="#7B1FA2", alpha=0.85)
    ax.bar(x, graph_w, width, label="Graph", color="#C62828", alpha=0.85)
    ax.bar(x + width, bm25_w, width, label="BM25", color="#00838F", alpha=0.85)

    ax.set_ylabel("Weight", fontsize=10)
    ax.set_title("Adaptive Retrieval Weights", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in query_types], fontsize=7)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.6)
    ax.grid(axis="y", alpha=0.3)

    # Right: Reranking weights
    ax = axes[1]
    features = ["Vector", "Graph", "BM25", "Lemma", "Morpho", "Compound", "Commentary", "Concept", "Centrality"]
    concept_short = [0.45, 0.20, 0.05, 0.12, 0.08, 0.03, 0.04, 0.02, 0.01]
    complex_long = [0.35, 0.15, 0.20, 0.12, 0.08, 0.04, 0.04, 0.01, 0.01]

    x = np.arange(len(features))
    width = 0.35

    ax.bar(x - width/2, concept_short, width, label="concept_short", color="#FF9933", alpha=0.85)
    ax.bar(x + width/2, complex_long, width, label="complex_long", color="#2196F3", alpha=0.85)

    ax.set_ylabel("Weight", fontsize=10)
    ax.set_title("Adaptive Re-ranking Weights", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=7, rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.55)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "weights.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def create_evaluation_chart():
    """Create evaluation results visualization with normalization comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: Unnormalized vs Normalized comparison
    ax = axes[0]
    datasets = ["HF Gita\nQA", "Kaggle\nGita QA", "Gita\nGuidance", "Edwin\nArnold", "ISKCON\nVedaBase"]
    unnorm = [0.6560, 0.4979, 0.4449, 0.3548, 0.3224]
    norm = [0.6802, 0.5014, 0.4437, 0.2916, 0.2591]

    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax.bar(x - width/2, unnorm, width, label="Unnormalized", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width/2, norm, width, label="MinMax Norm", color="#FF9933", alpha=0.85)

    for bar, score in zip(bars1, unnorm):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{score:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    for bar, score in zip(bars2, norm):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{score:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_ylabel("Avg Semantic Similarity", fontsize=10)
    ax.set_title("Normalization Comparison (5 Datasets)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=7)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.80)
    ax.grid(axis="y", alpha=0.3)

    # Right: Improvement chart
    ax = axes[1]
    improvements = [0.0242, 0.0035, -0.0012, -0.0632, -0.0633]
    colors = ["#4CAF50" if v > 0 else "#f44336" for v in improvements]

    bars = ax.bar(datasets, improvements, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, improvements):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + (0.002 if y > 0 else -0.005),
                f"{val:+.4f}", ha="center", va="bottom" if y > 0 else "top",
                fontsize=8, fontweight="bold")

    ax.set_ylabel("Semantic Similarity Change", fontsize=10)
    ax.set_title("MinMax Normalization Impact", fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="#666", linewidth=0.8, linestyle="-")
    ax.set_xticklabels(datasets, fontsize=7)
    ax.set_ylim(-0.08, 0.04)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "evaluation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def create_data_flow_diagram():
    """Create the data preprocessing flow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(5, 5.7, "Data Preprocessing Pipeline", fontsize=14, fontweight="bold",
            ha="center", va="center", color="#1a1a2e")

    boxes = [
        (1.0, 4.5, 1.8, 0.7, "dataset.xml\n(700 verses)", "#E3F2FD", "#1565C0"),
        (1.0, 3.3, 1.8, 0.7, "dataset.morpho.xml\n(Morphology)", "#E8F5E9", "#2E7D32"),
        (1.0, 2.1, 1.8, 0.7, "dataset.seg.xml\n(Lemmas)", "#FFF3E0", "#E65100"),
        (3.5, 3.3, 1.8, 0.7, "XML Parser\n(3 files)", "#F3E5F5", "#7B1FA2"),
        (5.5, 3.3, 1.8, 0.7, "Chunker\n(3507 chunks)", "#FFEBEE", "#C62828"),
        (7.5, 4.5, 1.8, 0.7, "FAISS Index\n(3507 vectors)", "#E0F7FA", "#00838F"),
        (7.5, 3.3, 1.8, 0.7, "BM25 Index\n(Lemma tokens)", "#FFF8E1", "#F57F17"),
        (7.5, 2.1, 1.8, 0.7, "Neo4j Graph\n(Concepts+Verses)", "#FCE4EC", "#AD1457"),
        (5.5, 1.5, 1.8, 0.7, "Graph Builder\n(Relationships)", "#E0F2F1", "#00695C"),
        (3.5, 1.5, 1.8, 0.7, "Concept Extractor\n(26 concepts)", "#E8EAF6", "#283593"),
    ]

    for x, y, w, h, text, fc, ec in boxes:
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.05", facecolor=fc,
                              edgecolor=ec, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, fontsize=7, ha="center", va="center",
                fontweight="bold", color="#1a1a2e")

    arrows = [
        (1.9, 4.5, 3.5, 3.7), (1.9, 3.3, 3.5, 3.3), (1.9, 2.1, 3.5, 2.9),
        (4.4, 3.3, 5.5, 3.3), (6.4, 3.6, 7.5, 4.5),
        (6.4, 3.3, 7.5, 3.3), (6.4, 3.0, 7.5, 2.1),
        (5.5, 1.9, 6.4, 2.1), (3.5, 1.9, 4.4, 2.1),
    ]

    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#455A64", lw=1.2))

    plt.tight_layout()
    path = OUTPUT_DIR / "data_flow.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def create_langgraph_diagram():
    """Create the LangGraph state machine diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(5, 3.7, "LangGraph State Machine", fontsize=14, fontweight="bold",
            ha="center", va="center", color="#1a1a2e")

    nodes = [
        (1.0, 2.0, 1.5, 0.7, "process_query", "#E3F2FD", "#1565C0"),
        (3.0, 2.0, 1.2, 0.7, "retrieve", "#E8F5E9", "#2E7D32"),
        (4.8, 2.0, 1.0, 0.7, "fuse", "#FFF3E0", "#E65100"),
        (6.4, 2.0, 1.0, 0.7, "rerank", "#F3E5F5", "#7B1FA2"),
        (8.2, 2.0, 1.2, 0.7, "generate", "#FFEBEE", "#C62828"),
    ]

    for x, y, w, h, text, fc, ec in nodes:
        ellipse = matplotlib.patches.Ellipse((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=2)
        ax.add_patch(ellipse)
        ax.text(x, y, text, fontsize=8, ha="center", va="center", fontweight="bold")

    arrows = [
        (1.75, 2.0, 2.4, 2.0), (3.6, 2.0, 4.3, 2.0),
        (5.3, 2.0, 5.9, 2.0), (6.9, 2.0, 7.6, 2.0),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#455A64", lw=1.5))

    # Loop arrow
    ax.annotate("", xy=(1.0, 2.5), xytext=(6.0, 2.5),
                arrowprops=dict(arrowstyle="->", color="#f44336", lw=1.5,
                                connectionstyle="arc3,rad=0.5"))
    ax.text(3.5, 3.2, "expand (confidence < 0.3, max 2 iterations)",
            fontsize=7, ha="center", va="center", color="#f44336", style="italic")

    # Decision diamond
    diamond = matplotlib.patches.FancyBboxPatch((5.9, 2.8), 1.0, 0.4,
                boxstyle="round,pad=0.05", facecolor="#FFF8E1", edgecolor="#F57F17", linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(6.4, 3.0, "confidence\ncheck", fontsize=6, ha="center", va="center")

    ax.annotate("", xy=(6.4, 2.7), xytext=(6.4, 2.35),
                arrowprops=dict(arrowstyle="<-", color="#F57F17", lw=1.2))

    plt.tight_layout()
    path = OUTPUT_DIR / "langgraph.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def build_pdf(arch_img, weights_img, eval_img, data_flow_img, langgraph_img):
    """Build the complete PDF report."""
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=20*mm,
        leftMargin=20*mm,
        topMargin=20*mm,
        bottomMargin=20*mm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"],
        fontSize=28, textColor=HexColor("#1a1a2e"),
        spaceAfter=6, alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=14, textColor=GRAY,
        spaceAfter=20, alignment=TA_CENTER,
    )
    heading1 = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontSize=18, textColor=HexColor("#1a1a2e"),
        spaceBefore=16, spaceAfter=8,
        borderPadding=(0, 0, 4, 0),
        borderWidth=0, borderColor=SAFFRON,
    )
    heading2 = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=14, textColor=HexColor("#CC7A00"),
        spaceBefore=12, spaceAfter=6,
    )
    heading3 = ParagraphStyle(
        "H3", parent=styles["Heading3"],
        fontSize=11, textColor=HexColor("#333333"),
        spaceBefore=8, spaceAfter=4,
    )
    body = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=14, textColor=HexColor("#333333"),
        spaceAfter=6, alignment=TA_JUSTIFY,
    )
    code_style = ParagraphStyle(
        "Code", parent=styles["Code"],
        fontSize=8, leading=10, textColor=HexColor("#1a1a2e"),
        backColor=HexColor("#f5f5f5"), borderPadding=6,
        spaceAfter=8,
    )
    caption_style = ParagraphStyle(
        "Caption", parent=styles["Normal"],
        fontSize=8, textColor=GRAY, alignment=TA_CENTER,
        spaceAfter=12, spaceBefore=4,
    )

    story = []

    # ── Title Page ──
    story.append(Spacer(1, 60))
    story.append(Paragraph("SRAG", title_style))
    story.append(Paragraph("Sanskrit RAG with Graph-Enhanced Linguistic Re-ranking", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="60%", thickness=2, color=SAFFRON))
    story.append(Spacer(1, 10))
    story.append(Paragraph("A Hybrid Retrieval-Augmented Generation System<br/>for the Bhagavad Gita", ParagraphStyle(
        "Sub2", parent=body, fontSize=12, alignment=TA_CENTER, textColor=GRAY)))
    story.append(Spacer(1, 30))

    meta_data = [
        ["Version", "0.2.0"],
        ["Pipeline", "LangGraph + Adaptive Re-ranking"],
        ["Embedding Model", "sanganaka/bge-m3-sanskritFT"],
        ["Generation Model", "MiMo v2.5"],
        ["Corpus", "3507 chunks across 18 chapters"],
        ["Tests", "48/48 passing"],
        ["Lint", "Ruff clean (0 errors)"],
    ]
    meta_table = Table(meta_data, colWidths=[120, 200])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), GRAY),
        ("TEXTCOLOR", (1, 0), (1, -1), HexColor("#333333")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, HexColor("#e0e0e0")),
    ]))
    story.append(meta_table)
    story.append(PageBreak())

    # ── Table of Contents ──
    story.append(Paragraph("Table of Contents", heading1))
    story.append(Spacer(1, 8))
    toc_items = [
        "1. Project Overview",
        "2. System Architecture",
        "3. Data Pipeline",
        "4. Retrieval System",
        "5. Adaptive Re-ranking",
        "6. LangGraph Orchestration",
        "7. Answer Generation",
        "8. Evaluation Results",
        "9. Project Structure",
        "10. Configuration",
        "11. API Reference",
        "12. Getting Started",
    ]
    for item in toc_items:
        story.append(Paragraph(item, ParagraphStyle(
            "TOC", parent=body, fontSize=11, spaceBefore=3, spaceAfter=3, leftIndent=20)))
    story.append(PageBreak())

    # ── 1. Project Overview ──
    story.append(Paragraph("1. Project Overview", heading1))
    story.append(Paragraph(
        "SRAG is a hybrid Retrieval-Augmented Generation system designed for the Bhagavad Gita. "
        "It combines three complementary retrieval methods (semantic vector search, knowledge graph traversal, "
        "and BM25 lexical matching) with a novel 9-feature adaptive re-ranking mechanism and LangGraph "
        "orchestration to generate scholarly, well-cited answers.", body))
    story.append(Paragraph(
        "The system processes queries in any language (English, Hindi, Sanskrit in IAST or Devanagari), "
        "converts them to IAST transliteration, extracts philosophical concepts (dharma, karma, yoga, etc.), "
        "and generates structured markdown answers with proper verse citations.", body))

    story.append(Paragraph("Key Innovations:", heading3))
    innovations = [
        "<b>Adaptive Retrieval:</b> Weights shift dynamically based on query type — vector-heavy for definitional queries, graph-heavy for factual queries, BM25-heavy for complex long queries.",
        "<b>9-Feature Re-ranking:</b> Combines vector score, graph centrality, BM25 score, lemma overlap, morphological match, compound match, commentary relevance, concept overlap, and graph centrality.",
        "<b>LangGraph State Machine:</b> Iterative query expansion when confidence is low (max 2 iterations), enabling self-correction.",
        "<b>Separation of Concerns:</b> Verses and commentaries are treated as separate content types — verses form the primary answer, commentaries appear only as brief scholarly context at the end.",
        "<b>Sanskrit-Aware Processing:</b> IAST-aware tokenization with suffix stripping, morphological feature extraction, and concept-aware query expansion.",
    ]
    for item in innovations:
        story.append(Paragraph(f"• {item}", ParagraphStyle(
            "Bullet", parent=body, leftIndent=20, firstLineIndent=-10)))
    story.append(PageBreak())

    # ── 2. System Architecture ──
    story.append(Paragraph("2. System Architecture", heading1))
    story.append(Paragraph(
        "The SRAG pipeline consists of five major stages: Query Processing, Hybrid Retrieval, "
        "Fusion, Re-ranking, and Generation. Each stage is modular and can be configured independently.", body))
    story.append(Spacer(1, 8))
    story.append(Image(str(arch_img), width=480, height=290))
    story.append(Paragraph("Figure 1: SRAG Pipeline Architecture", caption_style))

    story.append(Paragraph("2.1 Query Processing", heading2))
    story.append(Paragraph(
        "The QueryProcessor uses MiMo v2.5 to analyze user queries in any language. It performs: "
    "(1) language detection (English/Hindi/IAST/Devanagari), "
    "(2) IAST conversion via LLM, "
    "(3) concept extraction (26 Bhagavad Gita philosophical concepts), "
    "and (4) query type classification (concept_short, factual_short, complex_long, concept_medium, general_medium).", body))

    story.append(Paragraph("2.2 Hybrid Retrieval", heading2))
    story.append(Paragraph(
        "Three parallel retrieval strategies are executed simultaneously:", body))
    story.append(Paragraph(
        "• <b>Vector Search (FAISS):</b> Encodes the query in Devanagari using bge-m3-sanskritFT, "
        "searches the FAISS IndexFlatIP for the 50 most similar chunks by cosine similarity.", body))
    story.append(Paragraph(
        "• <b>Graph Search (Neo4j):</b> Executes full-text search on IAST text, concept neighborhood "
        "traversal, and combined queries. Returns up to 50 results with graph scores.", body))
    story.append(Paragraph(
        "• <b>BM25 Search:</b> Tokenizes the IAST query with suffix-aware lemma expansion, "
        "searches the BM25Okapi index for 50 matching chunks.", body))
    story.append(PageBreak())

    # ── 3. Data Pipeline ──
    story.append(Paragraph("3. Data Pipeline", heading1))
    story.append(Image(str(data_flow_img), width=480, height=240))
    story.append(Paragraph("Figure 2: Data Preprocessing Pipeline", caption_style))

    story.append(Paragraph("3.1 Source Corpus", heading2))
    story.append(Paragraph(
        "The Bhagavad Gita corpus consists of three XML files: (1) dataset.xml containing 700 verses "
        "in IAST and Devanagari with commentaries from three scholars (Sridhara Swamin — Advaita tradition, "
        "Visvanatha Chakravarti — Acintya-bhedabheda, Baladeva Vidyabhushana — Acintya-bhedabheda), "
        "(2) dataset.lemma-morphosyntax.xml with morphological annotations (case, gender, tense, mood) "
        "for each token, and (3) dataset.segmentation-lemma.xml with lemma-segmented text.", body))

    story.append(Paragraph("3.2 Chunking Strategy", heading2))
    story.append(Paragraph(
        "The XML parser extracts verses, commentaries, and morphological data. The chunker creates "
        "3507 chunks, each containing: IAST text, Devanagari text, lemmas (for BM25), morphological "
        "profile, concept associations, and chunk type (verse/commentary/combined). "
        "Verse chunks are prioritized over commentary chunks for better definitions.", body))

    story.append(Paragraph("3.3 Index Building", heading2))
    story.append(Paragraph(
        "<b>FAISS Index:</b> Uses sanganaka/bge-m3-sanskritFT (1024-dimensional Sanskrit embeddings). "
        "Encodes Devanagari text of all chunks. Built as IndexFlatIP for cosine similarity. "
        "Takes approximately 75 minutes on CPU for 3507 chunks.", body))
    story.append(Paragraph(
        "<b>BM25 Index:</b> Uses rank-bm25 library with IAST-aware tokenization. "
        "Query expansion strips Sanskrit suffixes (aḥ, am, ena, āya, etc.) and adds root + 'a' forms.", body))
    story.append(Paragraph(
        "<b>Neo4j Graph:</b> Nodes: Verse, Chapter, Concept, Commentator. "
        "Relationships: VERSE_OF_CHAPTER, HAS_CONCEPT, HAS_COMMENTARY, MENTIONS_CONCEPT. "
        "Full-text index verse_text_ft on text_iast field.", body))
    story.append(PageBreak())

    # ── 4. Retrieval System ──
    story.append(Paragraph("4. Retrieval System", heading1))
    story.append(Paragraph(
        "The HybridRetriever combines results from all three retrieval methods using "
        "Reciprocal Rank Fusion (RRF) with adaptive weights.", body))

    story.append(Paragraph("4.1 Reciprocal Rank Fusion", heading2))
    story.append(Paragraph(
        "RRF combines ranked lists from multiple retrievers into a single ranking. "
        "For each document, the RRF score is: <font face='Courier'>rrf_score = Σ (weight_i / (k + rank_i))</font>, "
        "where k = 60 (default). Documents appearing in multiple retrievers get boosted scores.", body))

    story.append(Paragraph("4.2 Adaptive Weights", heading2))
    story.append(Paragraph(
        "Weights shift based on query type. Short definitional queries favor vector similarity (0.50). "
        "Factual queries favor graph relationships (0.40). Complex long queries favor BM25 exact matching (0.40).", body))
    story.append(Spacer(1, 8))
    story.append(Image(str(weights_img), width=480, height=220))
    story.append(Paragraph("Figure 3: Adaptive Retrieval and Re-ranking Weights", caption_style))

    story.append(Paragraph("4.3 Score Propagation", heading2))
    story.append(Paragraph(
        "Each fused result carries individual scores from all three retrievers: "
        "vector_score, graph_score, bm25_score, and the combined rrf_score. "
        "This ensures downstream components can reason about the source of each result.", body))
    story.append(PageBreak())

    # ── 5. Adaptive Re-ranking ──
    story.append(Paragraph("5. Adaptive Re-ranking", heading1))
    story.append(Paragraph(
        "The LinguisticReranker extracts 9 features per candidate and applies dynamic weights "
        "based on query type. This is the core innovation of SRAG.", body))

    features_data = [
        ["#", "Feature", "Weight", "Description"],
        ["1", "score_vector", "0.40", "FAISS cosine similarity"],
        ["2", "score_graph", "0.20", "Neo4j graph retrieval score"],
        ["3", "score_bm25", "0.10", "BM25 lexical matching score"],
        ["4", "score_lemma", "0.12", "Query-document lemma overlap (Jaccard)"],
        ["5", "score_morph", "0.08", "Morphological case/gender match (Sanskrit suffix analysis)"],
        ["6", "score_compound", "0.03", "Bigram/compound match score"],
        ["7", "score_commentary", "0.04", "Commentary chunk relevance"],
        ["8", "score_concept", "0.02", "Concept overlap score"],
        ["9", "score_graph_centrality", "0.01", "Graph node degree / max_degree"],
    ]
    features_table = Table(features_data, colWidths=[20, 100, 50, 280])
    features_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (2, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e0e0e0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f9f9f9")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(features_table)
    story.append(Paragraph("Table 1: 9 Re-ranking Features and Default Weights", caption_style))

    story.append(Paragraph("5.1 Adaptive Profiles", heading2))
    profiles_data = [
        ["Query Type", "Vector", "Graph", "BM25", "Example"],
        ["concept_short", "0.45", "0.20", "0.05", "'What is dharma?'"],
        ["factual_short", "0.35", "0.30", "0.08", "'Who is Krishna?'"],
        ["complex_long", "0.35", "0.15", "0.20", "Long philosophical question"],
        ["concept_medium", "0.40", "0.22", "0.08", "'Explain karma yoga'"],
        ["general_medium", "0.40", "0.18", "0.12", "Default"],
    ]
    profiles_table = Table(profiles_data, colWidths=[80, 50, 50, 50, 180])
    profiles_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#CC7A00")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (3, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e0e0e0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f9f9f9")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(profiles_table)
    story.append(Paragraph("Table 2: Adaptive Weight Profiles by Query Type", caption_style))
    story.append(PageBreak())

    # ── 6. LangGraph Orchestration ──
    story.append(Paragraph("6. LangGraph Orchestration", heading1))
    story.append(Paragraph(
        "The SRAGGraphPipeline implements the full RAG pipeline as a LangGraph state machine "
        "with conditional routing for iterative query expansion.", body))
    story.append(Spacer(1, 8))
    story.append(Image(str(langgraph_img), width=480, height=190))
    story.append(Paragraph("Figure 4: LangGraph State Machine", caption_style))

    story.append(Paragraph("6.1 State Machine Nodes", heading2))
    nodes_data = [
        ["Node", "Function", "Input", "Output"],
        ["process_query", "IAST conversion + concept extraction", "query", "query_iast, concepts, query_type"],
        ["retrieve", "Vector + graph + BM25 parallel search", "query_iast, concepts", "vector_results, graph_results, bm25_results"],
        ["fuse", "Adaptive RRF fusion", "all results", "fused_results"],
        ["rerank", "9-feature linguistic re-ranking", "fused_results", "reranked_results, confidence"],
        ["generate", "MiMo v2.5 answer generation", "reranked_results", "answer, citations"],
    ]
    nodes_table = Table(nodes_data, colWidths=[75, 140, 100, 140])
    nodes_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e0e0e0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f9f9f9")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(nodes_table)
    story.append(Paragraph("Table 3: LangGraph State Machine Nodes", caption_style))

    story.append(Paragraph("6.2 Conditional Routing", heading2))
    story.append(Paragraph(
        "After the rerank node, a conditional edge checks the average reranking confidence. "
        "If confidence < 0.3 and iteration count < 2, the pipeline loops back to process_query "
        "with expanded concepts. Otherwise, it proceeds to generate.", body))

    story.append(Paragraph("6.3 Confidence Scoring", heading2))
    story.append(Paragraph(
        "Pipeline confidence is computed as: <font face='Courier'>overall = 0.3 × retrieval + 0.5 × reranking + 0.2 × generation</font>. "
        "Retrieval confidence uses sigmoid normalization. Generation confidence is based on citation count "
        "(1.0 if ≥ 3 citations, decreasing otherwise).", body))
    story.append(PageBreak())

    # ── 7. Answer Generation ──
    story.append(Paragraph("7. Answer Generation", heading1))
    story.append(Paragraph(
        "The AnswerGenerator uses MiMo v2.5 (via OpenAI-compatible API) to generate structured "
        "markdown answers. The prompt enforces an explanation-first structure.", body))

    story.append(Paragraph("7.1 Prompt Design", heading2))
    story.append(Paragraph(
        "The system prompt enforces 6 rules:", body))
    rules = [
        "<b>Explanation-first:</b> Model writes its own synthesis of retrieved verses, teaching the concept.",
        "<b>Verses as evidence:</b> Quote 1-2 lines, explain in plain language — never chain 'Verse X says...'.",
        "<b>Markdown formatting:</b> Headings, bold key terms, bullet points.",
        "<b>Commentary as appendix:</b> Single most relevant commentary at the end in 'Scholarly Context'.",
        "<b>Direct answers:</b> Answer fully without hedging.",
        "<b>Sanskrit notation:</b> IAST + Devanagari in parentheses + English meaning.",
    ]
    for rule in rules:
        story.append(Paragraph(f"• {rule}", ParagraphStyle(
            "Bullet", parent=body, leftIndent=20, firstLineIndent=-10)))

    story.append(Paragraph("7.2 Prompt Structure", heading2))
    prompt_parts = [
        ["Section", "Content", "Purpose"],
        ["User Question", "{query}", "Original user query"],
        ["Retrieved Verses", "Top-5 verses with IAST + Devanagari", "Primary source material"],
        ["Traditional Commentary", "Single highest-confidence commentary", "Supplementary context"],
    ]
    prompt_table = Table(prompt_parts, colWidths=[100, 180, 180])
    prompt_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e0e0e0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f9f9f9")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(prompt_table)
    story.append(Paragraph("Table 4: Prompt Structure", caption_style))

    story.append(Paragraph("7.3 Generation Settings", heading2))
    story.append(Paragraph(
        "• <b>Model:</b> MiMo v2.5 (mimo-v2.5, lowercase)<br/>"
        "• <b>API:</b> https://api.xiaomimimo.com/v1 (OpenAI-compatible)<br/>"
        "• <b>Temperature:</b> 0.3 (low creativity, high factual accuracy)<br/>"
        "• <b>Max tokens:</b> 2048<br/>"
        "• <b>Citation extraction:</b> Regex pattern BhG\\s+\\d+\\.\\d+", body))
    story.append(PageBreak())

    # ── 8. Evaluation Results ──
    story.append(Paragraph("8. Evaluation Results", heading1))
    story.append(Image(str(eval_img), width=480, height=220))
    story.append(Paragraph("Figure 5: Evaluation Results by Dataset and Query Type", caption_style))

    story.append(Paragraph("8.1 Datasets", heading2))
    datasets_info = [
        ["Dataset", "Type", "Samples", "Description"],
        ["Gita Guidance QA", "QA pairs", "711", "Modern life questions answered with Gita wisdom"],
        ["Edwin Arnold QA", "QA pairs", "500", "Factual questions from 'The Song Celestial'"],
        ["ISKCON VedaBase", "Verse commentary", "657", "Verse-by-verse Gaudiya Vaishnava commentary"],
    ]
    datasets_table = Table(datasets_info, colWidths=[90, 80, 50, 240])
    datasets_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e0e0e0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f9f9f9")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(datasets_table)
    story.append(Paragraph("Table 5: Evaluation Datasets", caption_style))

    story.append(Paragraph("8.2 Results (45 samples)", heading2))
    results_data = [
        ["Dataset", "Samples", "Semantic Sim.", "Word Overlap"],
        ["Gita Guidance QA", "21", "0.5013", "0.2781"],
        ["ISKCON VedaBase", "15", "0.3016", "—"],
        ["Edwin Arnold QA", "9", "0.2693", "0.2769"],
        ["Overall", "45", "0.3884", "—"],
    ]
    results_table = Table(results_data, colWidths=[110, 60, 80, 80])
    results_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#CC7A00")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e0e0e0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f9f9f9")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(results_table)
    story.append(Paragraph("Table 6: Evaluation Results Summary", caption_style))

    story.append(Paragraph("8.3 Analysis", heading2))
    story.append(Paragraph(
        "• <b>Gita Guidance QA (0.5013):</b> Best match — both SRAG and the dataset answer modern-life "
        "questions with Gita wisdom. The semantic alignment is strongest here.", body))
    story.append(Paragraph(
        "• <b>ISKCON VedaBase (0.3016):</b> Lower because ISKCON answers are from a single Gaudiya "
        "tradition, while SRAG uses 3 commentators (Advaita + Acintya-bhedabheda).", body))
    story.append(Paragraph(
        "• <b>Edwin Arnold QA (0.2693):</b> Hardest — short factual answers (2-10 words) are "
        "difficult for semantic similarity metrics to match against longer SRAG responses.", body))
    story.append(PageBreak())

    # ── 9. Project Structure ──
    story.append(Paragraph("9. Project Structure", heading1))
    structure = [
        ["Path", "Description"],
        ["configs/config.yaml", "All configuration (weights, profiles, paths, API settings)"],
        ["main.py", "CLI entry point (preprocess, build-graph, build-indices, query)"],
        ["api_server.py", "FastAPI backend with /api/query and /api/health endpoints"],
        ["evaluate.py", "Single-dataset evaluation pipeline"],
        ["evaluate_comprehensive.py", "3-dataset evaluation (Gita Guidance + Edwin Arnold + ISKCON)"],
        ["src/preprocessing/xml_parser.py", "XML corpus parsing (3 files)"],
        ["src/preprocessing/chunker.py", "Chunk creation and persistence (3507 chunks)"],
        ["src/preprocessing/concept_extractor.py", "26 Bhagavad Gita philosophical concepts"],
        ["src/preprocessing/iast_devanagari.py", "IAST ↔ Devanagari conversion"],
        ["src/preprocessing/morpho_extractor.py", "Morphological feature extraction"],
        ["src/preprocessing/graph_builder.py", "Neo4j graph construction"],
        ["src/retrieval/vector_store.py", "FAISS vector store with bge-m3-sanskritFT"],
        ["src/retrieval/bm25_retriever.py", "BM25Okapi with Sanskrit suffix-aware tokenization"],
        ["src/retrieval/graph_retriever.py", "Neo4j graph queries (full-text + concept)"],
        ["src/retrieval/hybrid_fusion.py", "Adaptive RRF + weighted fusion"],
        ["src/reranking/adaptive_reranker.py", "5 query type profiles with dynamic weights"],
        ["src/reranking/linguistic_reranker.py", "9-feature re-ranking with Sanskrit morpho hints"],
        ["src/reranking/feature_extractors.py", "Lemma, compound, concept, centrality extractors"],
        ["src/reranking/confidence.py", "Pipeline confidence scoring (sigmoid + Platt)"],
        ["src/generation/generator.py", "MiMo v2.5 answer generation"],
        ["src/generation/query_processor.py", "IAST conversion + concept extraction via MiMo"],
        ["src/generation/prompt_templates.py", "System + user prompt templates"],
        ["src/langchain_components/graph.py", "LangGraph state machine (5 nodes)"],
        ["src/langchain_components/state.py", "TypedDict state definition"],
        ["tests/", "48 tests across 4 modules"],
        ["web/", "React + Vite frontend (saffron/gold theme)"],
    ]
    struct_table = Table(structure, colWidths=[180, 280])
    struct_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e0e0e0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f9f9f9")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(struct_table)
    story.append(Paragraph("Table 7: Project File Structure", caption_style))
    story.append(PageBreak())

    # ── 10. Configuration ──
    story.append(Paragraph("10. Configuration", heading1))
    story.append(Paragraph(
        "All settings are centralized in <font face='Courier'>configs/config.yaml</font>. "
        "Key sections:", body))

    config_items = [
        "<b>data:</b> Paths for raw XML, processed chunks, FAISS index, graph import files",
        "<b>embedding:</b> Model name (bge-m3-sanskritFT), device (cpu), batch_size (8), max_length (256)",
        "<b>neo4j:</b> URI, credentials, database name",
        "<b>retrieval:</b> top_k (50), fusion_method (rrf), rrf_k (60), adaptive_weights (true)",
        "<b>reranking:</b> top_n (5), method (weighted), adaptive (true), 9 feature weights",
        "<b>adaptive_profiles:</b> 5 query type profiles with retrieval and reranking weights",
        "<b>generation:</b> provider (mimo), model (mimo-v2.5), temperature (0.3), max_tokens (2048)",
        "<b>langgraph:</b> enabled (true), max_iterations (2), confidence_threshold (0.3)",
        "<b>confidence:</b> pipeline_weights (retrieval: 0.3, reranking: 0.5, generation: 0.2)",
    ]
    for item in config_items:
        story.append(Paragraph(f"• {item}", ParagraphStyle(
            "Bullet", parent=body, leftIndent=20, firstLineIndent=-10)))

    story.append(PageBreak())

    # ── 11. API Reference ──
    story.append(Paragraph("11. API Reference", heading1))

    story.append(Paragraph("11.1 POST /api/query", heading2))
    api_req = [
        ["Field", "Type", "Description"],
        ["query", "string", "User query in any language"],
    ]
    api_req_table = Table(api_req, colWidths=[80, 60, 300])
    api_req_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e0e0e0")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(api_req_table)
    story.append(Paragraph("Table 8: Request Schema", caption_style))

    api_resp = [
        ["Field", "Type", "Description"],
        ["query", "string", "Original query"],
        ["query_iast", "string", "IAST transliteration"],
        ["query_devanagari", "string", "Devanagari rendering"],
        ["concepts", "list[str]", "Extracted concepts"],
        ["answer", "string", "Generated markdown answer"],
        ["verses_cited", "list[str]", "Verse references cited"],
        ["top_verses", "list[dict]", "Top 5 retrieved verses"],
        ["pipeline_confidence", "dict", "Confidence scores"],
        ["query_type", "string", "Detected query type"],
    ]
    api_resp_table = Table(api_resp, colWidths=[100, 60, 260])
    api_resp_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e0e0e0")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(api_resp_table)
    story.append(Paragraph("Table 9: Response Schema", caption_style))

    story.append(Paragraph("11.2 GET /api/health", heading2))
    story.append(Paragraph(
        "Returns: { status: 'ok', chunks: int, pipeline_type: 'langgraph' | 'standard' }", body))

    story.append(PageBreak())

    # ── 12. Getting Started ──
    story.append(Paragraph("12. Getting Started", heading1))

    story.append(Paragraph("12.1 Prerequisites", heading2))
    prereqs = [
        "Python 3.11+",
        "Neo4j 5.x running on bolt://localhost:7687",
        "MiMo API key (MIMO_API_KEY in .env)",
        "~4 GB RAM for embedding model",
    ]
    for p in prereqs:
        story.append(Paragraph(f"• {p}", body))

    story.append(Paragraph("12.2 Installation", heading2))
    story.append(Paragraph(
        "<font face='Courier'>pip install -r requirements.txt</font>", code_style))
    story.append(Paragraph(
        "<font face='Courier'>echo \"MIMO_API_KEY=your_key\" > .env</font>", code_style))

    story.append(Paragraph("12.3 Build Indices", heading2))
    steps = [
        "python main.py preprocess",
        "python main.py build-graph",
        "python main.py build-indices",
    ]
    for step in steps:
        story.append(Paragraph(f"<font face='Courier'>{step}</font>", code_style))

    story.append(Paragraph("12.4 Query", heading2))
    story.append(Paragraph(
        "<font face='Courier'>python main.py query --query \"What is dharma?\" --langgraph</font>", code_style))

    story.append(Paragraph("12.5 Run Tests", heading2))
    story.append(Paragraph(
        "<font face='Courier'>pytest tests/ -v</font>", code_style))
    story.append(Paragraph("Expected: 48 passed", body))

    story.append(Paragraph("12.6 Web UI", heading2))
    story.append(Paragraph(
        "<font face='Courier'>python api_server.py</font>", code_style))
    story.append(Paragraph("Open http://localhost:8000", body))

    # Build PDF
    doc.build(story)
    print(f"PDF saved to: {PDF_PATH}")
    return PDF_PATH


if __name__ == "__main__":
    print("Generating diagrams...")
    arch_img = create_architecture_diagram()
    print(f"  Architecture: {arch_img}")

    weights_img = create_retrieval_weights_chart()
    print(f"  Weights: {weights_img}")

    eval_img = create_evaluation_chart()
    print(f"  Evaluation: {eval_img}")

    data_flow_img = create_data_flow_diagram()
    print(f"  Data flow: {data_flow_img}")

    langgraph_img = create_langgraph_diagram()
    print(f"  LangGraph: {langgraph_img}")

    print("\nBuilding PDF report...")
    pdf_path = build_pdf(arch_img, weights_img, eval_img, data_flow_img, langgraph_img)
    print(f"\nDone! PDF: {pdf_path}")
    print(f"File size: {pdf_path.stat().st_size / 1024:.1f} KB")
