import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from pypdf import PdfReader


class PDFDocumentLoader:
    """Extracts, parses, and chunks text from a PDF file while preserving page provenance."""

    def __init__(self, pdf_path: Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found at: {self.pdf_path}. "
                f"Please ensure '{self.pdf_path.name}' is located in the project root."
            )
        self.reader = PdfReader(str(self.pdf_path))
        self.total_pages = len(self.reader.pages)
        self.pages_text: List[Dict[str, Any]] = []
        self._load_pages()

    def _load_pages(self):
        """Extracts text page by page."""
        self.pages_text = []
        for i, page in enumerate(self.reader.pages):
            raw_text = page.extract_text() or ""
            # Clean up excessive whitespace while preserving paragraph breaks
            cleaned = re.sub(r"\r\n|\r", "\n", raw_text)
            cleaned = re.sub(r"[ \t]+", " ", cleaned)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
            self.pages_text.append({
                "page_number": i + 1,
                "text": cleaned,
                "char_count": len(cleaned)
            })

    def get_page_text(self, page_number: int) -> str:
        """Returns the full text of a specific 1-indexed page."""
        if 1 <= page_number <= self.total_pages:
            return self.pages_text[page_number - 1]["text"]
        raise IndexError(f"Page number {page_number} out of range (1 to {self.total_pages})")

    def chunk_document(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150
    ) -> List[Dict[str, Any]]:
        """Splits document pages into overlapping semantic text chunks with metadata.

        Args:
            chunk_size: Target character length per chunk.
            chunk_overlap: Number of characters to overlap between adjacent chunks.

        Returns:
            A list of chunk dictionaries with 'id', 'text', 'page_number', and 'chunk_index'.
        """
        chunks: List[Dict[str, Any]] = []

        for page in self.pages_text:
            page_num = page["page_number"]
            text = page["text"]

            if not text.strip():
                continue

            # If page is shorter than chunk size, treat it as a single chunk
            if len(text) <= chunk_size:
                chunks.append({
                    "id": f"p{page_num}_c0",
                    "page_number": page_num,
                    "chunk_index": 0,
                    "text": text,
                    "char_count": len(text)
                })
                continue

            # Break text into paragraphs or sentences
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            current_chunk = ""
            chunk_idx = 0

            for para in paragraphs:
                if len(current_chunk) + len(para) + 2 <= chunk_size:
                    current_chunk = f"{current_chunk}\n\n{para}".strip()
                else:
                    if current_chunk:
                        chunks.append({
                            "id": f"p{page_num}_c{chunk_idx}",
                            "page_number": page_num,
                            "chunk_index": chunk_idx,
                            "text": current_chunk,
                            "char_count": len(current_chunk)
                        })
                        chunk_idx += 1
                        # Retain overlap from end of current chunk
                        overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
                        current_chunk = f"{overlap_text}\n\n{para}".strip()
                    else:
                        # Paragraph itself is longer than chunk_size, split by sliding window
                        start = 0
                        while start < len(para):
                            end = min(start + chunk_size, len(para))
                            piece = para[start:end].strip()
                            if piece:
                                chunks.append({
                                    "id": f"p{page_num}_c{chunk_idx}",
                                    "page_number": page_num,
                                    "chunk_index": chunk_idx,
                                    "text": piece,
                                    "char_count": len(piece)
                                })
                                chunk_idx += 1
                            if end >= len(para):
                                break
                            start += (chunk_size - chunk_overlap)
                        current_chunk = ""

            if current_chunk:
                chunks.append({
                    "id": f"p{page_num}_c{chunk_idx}",
                    "page_number": page_num,
                    "chunk_index": chunk_idx,
                    "text": current_chunk,
                    "char_count": len(current_chunk)
                })

        return chunks

    def get_summary_metadata(self) -> Dict[str, Any]:
        """Returns high-level metadata about the document."""
        total_chars = sum(p["char_count"] for p in self.pages_text)
        sample_snippets = []
        for p in self.pages_text[:min(3, self.total_pages)]:
            first_line = p["text"].split("\n")[0] if p["text"] else "Empty page"
            sample_snippets.append(f"Page {p['page_number']}: {first_line[:80]}")

        return {
            "filename": self.pdf_path.name,
            "total_pages": self.total_pages,
            "total_characters": total_chars,
            "page_previews": sample_snippets
        }
