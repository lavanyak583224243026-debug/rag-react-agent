from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def create_minimal_pdf(filepath: Path, pages: list) -> None:
    """Generates a valid multi-page PDF using pure Python stream formatting."""
    objects = []
    offsets = []

    # Object 1: Catalog
    objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

    # Object 2: Pages container
    kids_refs = [f"{i + 3} 0 R" for i in range(len(pages))]
    kids_str = " ".join(kids_refs)
    objects.append(f"2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {len(pages)} >>\nendobj\n")

    # Font object ID
    font_id = len(pages) + 3

    # Generate Page objects and Content streams
    content_objects = []
    next_id = font_id + 1

    for i, page_text in enumerate(pages):
        page_obj_id = i + 3
        content_obj_id = next_id
        next_id += 1

        # Format stream with PDF text operators
        lines = page_text.split("\n")
        stream_cmds = ["BT", "/F1 12 Tf", "50 750 Td", "16 TL"]
        for line in lines:
            safe_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream_cmds.append(f"({safe_line}) '")
        stream_cmds.append("ET")
        stream_data = "\n".join(stream_cmds)
        stream_len = len(stream_data.encode("latin-1"))

        objects.append(
            f"{page_obj_id} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_obj_id} 0 R >>\nendobj\n"
        )
        content_objects.append(
            f"{content_obj_id} 0 obj\n<< /Length {stream_len} >>\nstream\n{stream_data}\nendstream\nendobj\n"
        )

    # Font object
    font_obj = f"{font_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"

    all_objs = objects + [font_obj] + content_objects

    # Build entire file with xref
    out = "%PDF-1.4\n"
    pos = len(out.encode("latin-1"))
    xref_table = ["xref\n0 " + str(len(all_objs) + 1) + "\n0000000000 65535 f \n"]

    body = ""
    for obj_str in all_objs:
        xref_table.append(f"{pos:010d} 00000 n \n")
        body += obj_str
        pos += len(obj_str.encode("latin-1"))

    xref_str = "".join(xref_table)
    trailer = f"trailer\n<< /Size {len(all_objs) + 1} /Root 1 0 R >>\nstartxref\n{pos}\n%%EOF\n"

    with open(filepath, "wb") as f:
        f.write((out + body + xref_str + trailer).encode("latin-1"))


def generate_sample_rag_pdf(target_path: Path) -> Path:
    """Creates a sample multi-page PDF document about Autonomous Agentic AI and RAG."""
    page_1 = """PROJECT TITAN: NEXT-GENERATION AGENTIC AI ARCHITECTURE
Author: Deep Research AI Labs
Publication Date: October 2026
Version: 1.0-RC

1. Executive Summary
Agentic artificial intelligence represents a paradigm shift from passive predictive
models to autonomous systems capable of reasoning, planning, and executing actions.
Project Titan evaluates the synergy between Retrieval-Augmented Generation (RAG)
and Reasoning and Action (ReAct) architectures.

2. Problem Statement
Traditional LLMs suffer from knowledge cutoff and hallucinations when answering
questions about private or proprietary domain documents. Standard RAG pipelines
retrieve documents in a single shot, often pulling irrelevant or incomplete chunks
without verifying if the extracted evidence actually answers the user's inquiry.
By augmenting RAG with ReAct loops, agents iteratively inspect, verify, and cite
factual evidence directly from source PDFs."""

    page_2 = """3. System Architecture & Methodology

3.1 The ReAct Reasoning Loop
The system utilizes a structured Thought-Action-Observation cycle:
- Thought: The agent analyzes the user prompt, determines what data is needed,
  and plans its next inquiry.
- Action: The agent invokes dedicated document tools: search_document, read_page,
  or get_document_info.
- Observation: The retrieval tool returns relevant snippets along with page attribution.
This loop executes up to 6 iterations until factual evidence is verified.

3.2 Dense Embedding and Retrieval Specifications
- Embedding Engine: Google Gemini gemini-embedding-001 (768 dimensions).
- Chunking Strategy: Sliding window of 800 characters with 150-character overlap.
- Hybrid Scoring: Dense cosine similarity combined with a +0.05 lexical match bonus.
- Cache Mechanism: SHA-256 content-hashed local cache to ensure sub-millisecond retrieval
  on subsequent queries."""

    page_3 = """4. Experimental Results and Benchmarks

We evaluated the Titan ReAct Agent across 500 complex multi-hop enterprise queries:
- Retrieval Precision@3: 94.8% (up from 78.2% in naive single-shot RAG).
- Hallucination Rate: Reduced from 14.6% to 0.4% under strict citation constraints.
- Multi-page Synthesis Success: 91.5% accuracy when reasoning across disparate pages.
- Average Latency per Query: 1.8 seconds using Gemini 3.8 Flash.

5. Key Conclusions & Recommendations
1. ReAct agents significantly outperform single-shot RAG in factual faithfulness.
2. Direct page-level citations enable 100% auditable provenance for enterprise compliance.
3. Hybrid dense-lexical scoring is critical for capturing exact technical codes and numbers."""

    create_minimal_pdf(target_path, [page_1, page_2, page_3])
    print(f"✅ Generated sample PDF: {target_path} (3 pages)")
    return target_path


if __name__ == "__main__":
    from config import PDF_PATH
    generate_sample_rag_pdf(PDF_PATH)
