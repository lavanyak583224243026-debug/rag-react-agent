import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import PDF_PATH
from pdf_loader import PDFDocumentLoader
from vector_store import LocalVectorStore
from rag_tools import RAGTools


def run_tests():
    print("Testing PDF Loader...")
    loader = PDFDocumentLoader(PDF_PATH)
    assert loader.total_pages == 3, f"Expected 3 pages, got {loader.total_pages}"
    chunks = loader.chunk_document()
    assert len(chunks) > 0, "Expected non-empty chunks"
    print(f"  [PASS] Extracted {loader.total_pages} pages into {len(chunks)} chunks.")

    print("Testing Vector Store & Lexical Search...")
    vs = LocalVectorStore(client=None)
    vs.build_or_load_index(PDF_PATH, chunks)
    res = vs.search("hallucination rate", top_k=2)
    assert len(res) > 0, "Expected at least 1 search result"
    print(f"  [PASS] Top search hit page: {res[0]['page_number']}")

    print("Testing RAG Tools...")
    tools = RAGTools(loader, vs)
    info = tools.get_document_info()
    assert "Total Pages: 3" in info
    print(f"  [PASS] get_document_info() verified.")

    page2_text = tools.read_page(2)
    assert "System Architecture" in page2_text or "ReAct Reasoning Loop" in page2_text
    print(f"  [PASS] read_page(2) verified.")

    print("\nAll unit tests passed successfully!")


if __name__ == "__main__":
    run_tests()
