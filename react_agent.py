import sys
from typing import List, Dict, Any
from colorama import Fore, Style, init
from google import genai
from google.genai import types

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rag_tools import RAGTools

init(autoreset=True)


REACT_SYSTEM_INSTRUCTION = """
You are an Agentic AI Developer assistant operating in a strict ReAct paradigm.

Your mission is to answer the user's inquiry based exclusively on the provided PDF document.

Available tools:

1. search_document(query, top_k)
2. read_page(page_number)
3. get_document_info()

Before answering, use the tools when document information is needed.

After receiving tool results, analyze them and provide a final answer.

Always cite specific page numbers such as [Page 1] or [Page 2].

If the answer is not present in the PDF, clearly say that the document does not contain the requested information.
"""


class ReActAgent:
    """Agentic ReAct RAG agent using Gemini."""

    def __init__(
        self,
        client: genai.Client,
        rag_tools: RAGTools,
        model_name: str = "gemini-3.8-flash",
        max_steps: int = 6,
        verbose: bool = True
    ):
        self.client = client
        self.rag_tools = rag_tools
        self.model_name = model_name
        self.max_steps = max_steps
        self.verbose = verbose

        # IMPORTANT:
        # Do NOT pass bound Python methods directly to Gemini.
        # Use function declarations instead.
        self.tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="search_document",
                        description="Search the PDF document for relevant information.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "query": types.Schema(
                                    type="STRING",
                                    description="The search query."
                                ),
                                "top_k": types.Schema(
                                    type="INTEGER",
                                    description="Number of results to retrieve."
                                ),
                            },
                            required=["query"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="read_page",
                        description="Read the complete text of a specific PDF page.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "page_number": types.Schema(
                                    type="INTEGER",
                                    description="PDF page number."
                                ),
                            },
                            required=["page_number"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="get_document_info",
                        description="Get information about the PDF document.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={},
                        ),
                    ),
                ]
            )
        ]

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:

        if name == "search_document":
            query = args.get("query", "")
            top_k = int(args.get("top_k", 3))

            return self.rag_tools.search_document(
                query=query,
                top_k=top_k
            )

        elif name == "read_page":
            page_number = int(args.get("page_number", 1))

            return self.rag_tools.read_page(
                page_number=page_number
            )

        elif name == "get_document_info":
            return self.rag_tools.get_document_info()

        return f"Error: Unknown tool '{name}'."

    def _call_model(self, contents: List[types.Content], config: types.GenerateContentConfig):
        """Generates content with automatic fallback to high-quota models on 429 quota exhaustion."""
        models_to_try = [self.model_name]
        for fallback in ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]:
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        last_error = None
        for current_model in models_to_try:
            try:
                resp = self.client.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config=config
                )
                if current_model != self.model_name:
                    self.model_name = current_model  # Stick with working model
                return resp
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    if self.verbose:
                        print(f"{Fore.YELLOW}⚠️ Model '{current_model}' hit free tier quota limit. Switching to '{models_to_try[models_to_try.index(current_model)+1] if models_to_try.index(current_model)+1 < len(models_to_try) else 'fallback'}'...{Style.RESET_ALL}")
                    last_error = e
                    continue
                raise e
        raise last_error

    def run(self, user_query: str) -> Dict[str, Any]:

        if self.verbose:
            print(f"\n{Fore.MAGENTA}{'=' * 65}{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}🤖 ReAct Agent Activated{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Question: {user_query}{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}{'=' * 65}{Style.RESET_ALL}\n")

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=user_query
                    )
                ]
            )
        ]

        config = types.GenerateContentConfig(
            system_instruction=REACT_SYSTEM_INSTRUCTION,
            tools=self.tools,
            temperature=0.2,
        )

        traces = []
        step_count = 0

        while step_count < self.max_steps:

            step_count += 1

            if self.verbose:
                print(
                    f"{Fore.CYAN}--- Step "
                    f"{step_count} of {self.max_steps} ---"
                    f"{Style.RESET_ALL}"
                )

            try:
                response = self._call_model(contents=contents, config=config)
            except Exception as e:

                err_msg = f"API Generation Error: {e}"

                if self.verbose:
                    print(
                        f"{Fore.RED}❌ {err_msg}"
                        f"{Style.RESET_ALL}"
                    )

                return {
                    "answer": err_msg,
                    "steps": step_count,
                    "traces": traces
                }

            if not response.candidates:
                return {
                    "answer": "No response received from Gemini.",
                    "steps": step_count,
                    "traces": traces
                }

            candidate = response.candidates[0]

            parts = (
                candidate.content.parts
                if candidate.content
                else []
            )

            thought_text = ""
            function_calls = []

            for part in parts:

                if getattr(part, "text", None):
                    thought_text += part.text

                if getattr(part, "function_call", None):
                    function_calls.append(
                        part.function_call
                    )

            if thought_text.strip():

                clean_thought = thought_text.strip()

                if self.verbose:
                    print(
                        f"{Fore.CYAN}🧠 Thought:"
                        f"{Style.RESET_ALL} "
                        f"{clean_thought}\n"
                    )

                traces.append({
                    "type": "thought",
                    "content": clean_thought
                })

            # -------------------------------------------------
            # TOOL CALL
            # -------------------------------------------------

            if function_calls:

                contents.append(candidate.content)

                response_parts = []

                for call in function_calls:

                    call_name = call.name
                    call_args = dict(call.args or {})

                    args_str = ", ".join(
                        f"{k}={repr(v)}"
                        for k, v in call_args.items()
                    )

                    if self.verbose:
                        print(
                            f"{Fore.YELLOW}⚡ Action:"
                            f"{Style.RESET_ALL} "
                            f"{call_name}({args_str})"
                        )

                    traces.append({
                        "type": "action",
                        "tool": call_name,
                        "args": call_args
                    })

                    # Execute locally
                    observation = self._execute_tool(
                        call_name,
                        call_args
                    )

                    if self.verbose:

                        obs_preview = (
                            observation
                            if len(observation) < 600
                            else observation[:600]
                            + "... [truncated]"
                        )

                        print(
                            f"{Fore.GREEN}👁️ Observation:"
                            f"{Style.RESET_ALL}\n"
                            f"{obs_preview}\n"
                        )

                    traces.append({
                        "type": "observation",
                        "content": observation
                    })

                    # Send result back to Gemini
                    response_parts.append(
                        types.Part.from_function_response(
                            name=call_name,
                            response={
                                "result": observation
                            }
                        )
                    )

                contents.append(
                    types.Content(
                        role="user",
                        parts=response_parts
                    )
                )

            else:

                # -------------------------------------------------
                # FINAL ANSWER
                # -------------------------------------------------

                final_answer = thought_text.strip()

                if not final_answer:
                    final_answer = (
                        response.text
                        or "Unable to generate an answer."
                    )

                if self.verbose:

                    print(
                        f"{Fore.GREEN}"
                        f"{'=' * 65}"
                        f"{Style.RESET_ALL}"
                    )

                    print(
                        f"{Fore.GREEN}🎯 Final Answer:"
                        f"{Style.RESET_ALL}"
                    )

                    print(final_answer)

                    print(
                        f"{Fore.GREEN}"
                        f"{'=' * 65}"
                        f"{Style.RESET_ALL}\n"
                    )

                return {
                    "answer": final_answer,
                    "steps": step_count,
                    "traces": traces
                }

        # ---------------------------------------------------------
        # MAX STEPS REACHED
        # ---------------------------------------------------------

        if self.verbose:
            print(
                f"{Fore.YELLOW}"
                f"⚠️ Reached maximum step limit ({self.max_steps}). Synthesizing final answer from observations..."
                f"{Style.RESET_ALL}\n"
            )

        try:
            synthesis_contents = list(contents) + [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(
                        text="Based on all the observations gathered from the document so far, please provide your final answer now. Cite specific page numbers if relevant."
                    )]
                )
            ]
            final_resp = self._call_model(
                contents=synthesis_contents,
                config=types.GenerateContentConfig(
                    system_instruction=REACT_SYSTEM_INSTRUCTION,
                    temperature=0.2
                )
            )
            final_text = ""
            if final_resp and final_resp.candidates:
                for part in final_resp.candidates[0].content.parts:
                    if getattr(part, "text", None):
                        final_text += part.text

            final_answer = final_text.strip() if final_text.strip() else "No conclusive answer could be formed from the document."

            if self.verbose:
                print(
                    f"{Fore.GREEN}"
                    f"{'=' * 65}\n"
                    f"🎯 Final Answer:\n"
                    f"{final_answer}\n"
                    f"{'=' * 65}"
                    f"{Style.RESET_ALL}\n"
                )

            return {
                "answer": final_answer,
                "steps": step_count,
                "traces": traces
            }
        except Exception as e:
            err_answer = f"Max steps reached. Could not synthesize: {e}"
            if self.verbose:
                print(f"{Fore.RED}{err_answer}{Style.RESET_ALL}")
            return {
                "answer": err_answer,
                "steps": step_count,
                "traces": traces
            }