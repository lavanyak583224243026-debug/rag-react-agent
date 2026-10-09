import json
from typing import Optional, Dict, Any, List
from pdf_loader import PDFDocumentLoader
from vector_store import LocalVectorStore


class RAGTools:
    """Encapsulates RAG tools used by the ReAct agent to interact with the PDF document."""

    def __init__(self, pdf_loader: PDFDocumentLoader, vector_store: LocalVectorStore):
        self.pdf_loader = pdf_loader
        self.vector_store = vector_store

    def search_document(self, query: str, top_k: int = 3) -> str:
        """Searches the PDF document for relevant passages based on semantic and keyword similarity.

        Args:
            query: The search question or keywords to look for.
            top_k: Number of most relevant passages to return (default is 3).

        Returns:
            A formatted string of relevant passages including page numbers and relevance scores.
        """
        results = self.vector_store.search(query=query, top_k=top_k)
        if not results:
            return "No relevant passages found in the document for the query."

        output_lines = [f"Found {len(results)} relevant passages for '{query}':\n"]
        for i, item in enumerate(results, 1):
            score_pct = f"{item['score']:.2f}"
            output_lines.append(
                f"[Passage {i} | Page {item['page_number']} | Relevance: {score_pct}]\n"
                f"{item['text']}\n"
            )
        return "\n".join(output_lines)

    def read_page(self, page_number: int) -> str:
        """Reads the complete text of a specific page from the PDF document.

        Args:
            page_number: The 1-based index of the page to read.

        Returns:
            The complete text content of the requested page.
        """
        try:
            text = self.pdf_loader.get_page_text(page_number)
            if not text.strip():
                return f"Page {page_number} is blank or contains no extractable text."
            return f"--- Content of Page {page_number} (Total chars: {len(text)}) ---\n\n{text}"
        except IndexError:
            return f"Error: Page number {page_number} is out of bounds. The document has {self.pdf_loader.total_pages} pages."
        except Exception as e:
            return f"Error reading page {page_number}: {str(e)}"

    def get_document_info(self) -> str:
        """Retrieves high-level metadata about the loaded PDF document including total pages and preview.

        Returns:
            A summary of document metadata, total pages, character count, and sample headings.
        """
        meta = self.pdf_loader.get_summary_metadata()
        previews = "\n".join(f"  - {p}" for p in meta["page_previews"])
        return (
            f"Document: {meta['filename']}\n"
            f"Total Pages: {meta['total_pages']}\n"
            f"Total Characters: {meta['total_characters']}\n"
            f"Page previews:\n{previews}"
        )
