import sys
import argparse
from pathlib import Path
from colorama import Fore, Style, init
from google import genai

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import (
    BASE_DIR,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    EMBEDDING_MODEL,
    PDF_PATH,
    CACHE_DIR,
    validate_api_key,
)
from pdf_loader import PDFDocumentLoader
from vector_store import LocalVectorStore
from rag_tools import RAGTools
from react_agent import ReActAgent
from create_sample_pdf import generate_sample_rag_pdf

init(autoreset=True)


def ensure_pdf_exists(pdf_path: Path) -> bool:
    """Verifies that the PDF file exists; offers to generate a sample if missing."""
    if pdf_path.exists():
        return True

    print(f"\n{Fore.YELLOW}⚠️  Notice: '{pdf_path.name}' was not found in:{Style.RESET_ALL}")
    print(f"   {pdf_path}")
    print(f"\n{Fore.CYAN}Generating a sample 3-page research paper PDF for testing...{Style.RESET_ALL}")
    try:
        generate_sample_rag_pdf(pdf_path)
        print(f"{Fore.GREEN}✅ Successfully generated '{pdf_path.name}'. You can replace this file anytime with your own PDF!{Style.RESET_ALL}\n")
        return True
    except Exception as e:
        print(f"{Fore.RED}❌ Failed to create sample PDF: {e}{Style.RESET_ALL}")
        return False


def setup_rag_pipeline(pdf_path: Path, api_key: str, force_rebuild: bool = False):
    """Initializes the PDF loader, Vector Store, RAG Tools, and Gemini Client."""
    print(f"{Fore.CYAN}📄 Loading PDF document: {pdf_path.name}...{Style.RESET_ALL}")
    loader = PDFDocumentLoader(pdf_path)
    chunks = loader.chunk_document(chunk_size=800, chunk_overlap=150)
    print(f"   Pages: {loader.total_pages} | Chunks created: {len(chunks)}")

    # Initialize Gemini client
    client = None
    if api_key and api_key != "YOUR_GEMINI_API_KEY_HERE":
        client = genai.Client(api_key=api_key)

    # Initialize Vector Store
    vector_store = LocalVectorStore(
        client=client,
        model_name=EMBEDDING_MODEL,
        cache_dir=CACHE_DIR
    )
    vector_store.build_or_load_index(pdf_path=pdf_path, chunks=chunks, force_rebuild=force_rebuild)

    # Package tools
    rag_tools = RAGTools(pdf_loader=loader, vector_store=vector_store)
    return loader, vector_store, rag_tools, client


def run_diagnostic_check(pdf_path: Path):
    """Performs a comprehensive diagnostic health check of the RAG system."""
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🔍 RUNNING RAG REACT SYSTEM DIAGNOSTIC CHECK{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

    # 1. Check .env and API key
    key = validate_api_key()
    if key:
        masked_key = key[:6] + "..." + key[-4:] if len(key) > 10 else "***"
        print(f"✅ Gemini API Key: Configured ({masked_key})")
    else:
        print(f"⚠️  Gemini API Key: Placeholder or missing in .env")

    # 2. Check PDF document
    if pdf_path.exists():
        print(f"✅ PDF Document: Present ({pdf_path.name}, {pdf_path.stat().st_size} bytes)")
    else:
        print(f"⚠️  PDF Document: Not found at {pdf_path.name} (will auto-generate sample if requested)")

    # 3. Test PDF extraction
    ensure_pdf_exists(pdf_path)
    try:
        loader = PDFDocumentLoader(pdf_path)
        chunks = loader.chunk_document()
        meta = loader.get_summary_metadata()
        print(f"✅ PDF Parser: Working ({meta['total_pages']} pages, {len(chunks)} chunks, {meta['total_characters']} chars)")
    except Exception as e:
        print(f"❌ PDF Parser Error: {e}")

    # 4. Check Models configuration
    print(f"✅ Reasoning Model: {GEMINI_MODEL}")
    print(f"✅ Embedding Model: {EMBEDDING_MODEL}")
    print(f"\n{Fore.GREEN}Diagnostic check complete.{Style.RESET_ALL}\n")


def interactive_loop(agent: ReActAgent, rag_tools: RAGTools):
    """Runs an interactive conversational loop for asking questions to the agent."""
    print(f"\n{Fore.GREEN}{'='*65}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}🌟 RAG ReAct Agent CLI Ready!{Style.RESET_ALL}")
    print(f"Ask any question about '{PDF_PATH.name}'.")
    print(f"Commands: 'info' (view doc stats) | 'exit' or 'quit' to terminate.")
    print(f"{Fore.GREEN}{'='*65}{Style.RESET_ALL}\n")

    while True:
        try:
            query = input(f"{Fore.CYAN}User > {Style.RESET_ALL}").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print(f"{Fore.YELLOW}Exiting RAG ReAct Agent. Goodbye!{Style.RESET_ALL}")
                break
            if query.lower() == "info":
                print(f"\n{rag_tools.get_document_info()}\n")
                continue

            # Execute ReAct Agent
            agent.run(query)

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Session interrupted. Exiting...{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"{Fore.RED}An error occurred: {e}{Style.RESET_ALL}\n")


def main():
    parser = argparse.ArgumentParser(description="Agentic ReAct RAG with Google Gemini")
    parser.add_argument("--query", "-q", type=str, help="Run a single question and exit")
    parser.add_argument("--check", action="store_true", help="Run system diagnostics and exit")
    parser.add_argument("--reindex", action="store_true", help="Force rebuild of vector embeddings cache")
    args = parser.parse_args()

    if args.check:
        run_diagnostic_check(PDF_PATH)
        return

    # Check PDF file
    if not ensure_pdf_exists(PDF_PATH):
        sys.exit(1)

    # Validate API key
    api_key = validate_api_key()
    if not api_key:
        print(f"{Fore.YELLOW}Please configure your GEMINI_API_KEY in .env before running queries.{Style.RESET_ALL}")
        sys.exit(1)

    # Setup RAG pipeline
    loader, vector_store, rag_tools, client = setup_rag_pipeline(
        pdf_path=PDF_PATH,
        api_key=api_key,
        force_rebuild=args.reindex
    ) 

    # Initialize ReAct Agent
    agent = ReActAgent(
        client=client,
        rag_tools=rag_tools,
        model_name=GEMINI_MODEL,
        max_steps=6,
        verbose=True
    )

    if args.query:
        agent.run(args.query)
    else:
        interactive_loop(agent, rag_tools)


if __name__ == "__main__":
    main()
