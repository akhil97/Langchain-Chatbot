import hashlib
import os

import streamlit as st
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openrouter import ChatOpenRouter

from app_config import (
    MAX_PDF_UPLOAD_MB,
    MAX_TOOL_CALLS,
    PDF_CHUNK_OVERLAP,
    PDF_CHUNK_SIZE,
    PDF_EMBEDDING_MODEL,
)
from rag_services import (
    assess_document_support,
    collect_activity,
    collect_contexts,
    collect_sources,
    create_search_tools,
    get_system_prompt,
    load_pdf_documents,
    load_or_build_pdf_hybrid_index,
    message_text,
    retrieve_pdf_context,
)
from ui_components import (
    render_activity,
    render_deepeval_evaluation,
    render_pdf_evidence,
    render_ragas_evaluation,
)


def main() -> None:
    st.set_page_config(page_title="Agentic RAG", page_icon="🔎", layout="wide")
    st.markdown(
        """
        <style>
        .block-container { max-width: 980px; padding-top: 2rem; }
        [data-testid="stChatMessage"] { border-radius: 14px; }
        [data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.2); }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Agentic RAG")
    st.caption(
        "Ask a question. Uploaded PDFs use hybrid keyword and semantic search; "
        "the agent can also search the web, Wikipedia, and arXiv."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("Settings")
        openrouter_api_key = st.text_input(
            "OpenRouter API key",
            value=os.getenv("OPENROUTER_API_KEY", ""),
            type="password",
            help="Used only for this session. It is not shown in chat output.",
        )
        tavily_api_key = st.text_input(
            "Tavily API key (optional)",
            value=os.getenv("TAVILY_API_KEY", ""),
            type="password",
            help="Needed only when the agent chooses web search.",
        )
        model_name = st.text_input("OpenRouter model", value="openai/gpt-4o-mini")
        embedding_model = st.text_input(
            "OpenRouter embedding model",
            value="openai/text-embedding-3-small",
            help="Used for RAGAS answer relevancy only. PDF retrieval uses a local embedding model.",
        )
        uploaded_pdfs = st.file_uploader(
            "Upload a folder of PDFs",
            type=["pdf"],
            accept_multiple_files="directory",
            max_upload_size=MAX_PDF_UPLOAD_MB,
            help=(
                f"Choose a folder; its PDF files will be indexed together. "
                f"Each file can be up to {MAX_PDF_UPLOAD_MB} MB."
            ),
        )
        if uploaded_pdfs:
            st.caption(f"{len(uploaded_pdfs)} PDF file(s) selected")

        st.divider()
        st.subheader("Conversation history")
        if st.session_state.chat_history:
            for turn_index, item in enumerate(st.session_state.chat_history):
                if item["role"] == "user":
                    st.markdown(f"**You:** {item['content'][:120]}")
                else:
                    st.caption(f"Agent: {item['content'][:120]}")
        else:
            st.caption("Your questions and follow-ups will appear here.")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        st.divider()
        st.caption("Tools: Tavily · Wikipedia · arXiv")
        st.caption(f"Maximum tool calls per answer: {MAX_TOOL_CALLS}")

    uploaded_pdfs = uploaded_pdfs or []
    pdf_digest = hashlib.sha256()
    for uploaded_pdf in sorted(uploaded_pdfs, key=lambda file: file.name):
        pdf_digest.update(uploaded_pdf.name.encode("utf-8"))
        pdf_digest.update(uploaded_pdf.getvalue())
    pdf_key = pdf_digest.hexdigest() if uploaded_pdfs else ""
    cached_pdf = st.session_state.get("pdf_documents_cache")
    if uploaded_pdfs and cached_pdf and cached_pdf["key"] == pdf_key:
        pdf_documents = cached_pdf["documents"]
        pdf_was_truncated = cached_pdf["truncated"]
        pdf_error = cached_pdf["error"]
    else:
        pdf_documents, pdf_was_truncated, pdf_error = load_pdf_documents(uploaded_pdfs)
        if uploaded_pdfs:
            st.session_state.pdf_documents_cache = {
                "key": pdf_key,
                "documents": pdf_documents,
                "truncated": pdf_was_truncated,
                "error": pdf_error,
            }
    if pdf_error:
        st.warning(pdf_error)
    if pdf_was_truncated:
        st.warning(
            "The PDF exceeds the local text extraction limit. Answers use only chunks "
            "from the extracted portion; missing content is not proof that the document "
            "does not contain it."
        )

    pdf_vector_store = None
    pdf_keyword_retriever = None
    pdf_index_error = None
    if uploaded_pdfs and pdf_documents:
        index_fingerprint = (
            f"{pdf_key}:{PDF_EMBEDDING_MODEL}:{PDF_CHUNK_SIZE}:{PDF_CHUNK_OVERLAP}"
        )
        document_key = hashlib.sha256(index_fingerprint.encode("utf-8")).hexdigest()
        cached_index = st.session_state.get("pdf_index_cache")
        if cached_index and cached_index["key"] == document_key:
            pdf_vector_store = cached_index["vector_store"]
            pdf_keyword_retriever = cached_index["keyword_retriever"]
        else:
            progress_bar = st.progress(0, text="Preparing the PDF hybrid index...")

            def update_index_progress(fraction: float, message: str) -> None:
                progress_bar.progress(fraction, text=message)

            try:
                (
                    pdf_vector_store,
                    pdf_keyword_retriever,
                    cache_hit,
                    chunk_count,
                ) = load_or_build_pdf_hybrid_index(
                    pdf_documents,
                    document_key,
                    progress_callback=update_index_progress,
                )
                st.session_state.pdf_index_cache = {
                    "key": document_key,
                    "vector_store": pdf_vector_store,
                    "keyword_retriever": pdf_keyword_retriever,
                }
                if cache_hit:
                    progress_bar.progress(
                        1.0,
                        text=f"Loaded cached hybrid index ({chunk_count:,} chunks)",
                    )
                else:
                    progress_bar.progress(
                        1.0,
                        text=f"PDF hybrid index ready ({chunk_count:,} chunks)",
                    )
            except Exception as exc:
                pdf_index_error = (
                    f"PDF indexing failed ({type(exc).__name__}). "
                    "Check that the local embedding model can be downloaded and "
                    "that the uploaded PDFs contain extractable text."
                )
                st.error(pdf_index_error)

    for turn_index, item in enumerate(st.session_state.chat_history):
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant":
                render_activity(item.get("activity", []))
                render_pdf_evidence(item.get("pdf_context", ""))
                if item.get("question"):
                    render_deepeval_evaluation(
                        item,
                        turn_index,
                        openrouter_api_key,
                        model_name,
                    )
                    render_ragas_evaluation(
                        item,
                        turn_index,
                        openrouter_api_key,
                        model_name,
                        embedding_model,
                    )

    question = st.chat_input("Ask a question...")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    if not openrouter_api_key.strip():
        answer = "Add your OpenRouter API key in the sidebar to start a conversation."
        with st.chat_message("assistant"):
            st.error(answer)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "activity": []}
        )
        return
    if not model_name.strip():
        answer = "Enter an OpenRouter model that supports tool calling."
        with st.chat_message("assistant"):
            st.error(answer)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "activity": []}
        )
        return

    pdf_context = ""
    pdf_contexts = []
    if uploaded_pdfs:
        if not pdf_documents:
            answer = "The selected folder has no extractable PDF text to index."
            with st.chat_message("assistant"):
                st.error(answer)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "activity": []}
            )
            return
        if pdf_index_error:
            with st.chat_message("assistant"):
                st.error(pdf_index_error)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": pdf_index_error, "activity": []}
            )
            return
        try:
            with st.spinner("Searching PDFs by meaning and keywords..."):
                pdf_context, pdf_contexts = retrieve_pdf_context(
                    pdf_vector_store,
                    pdf_keyword_retriever,
                    question,
                )
        except Exception as exc:
            answer = (
                f"PDF indexing or retrieval failed ({type(exc).__name__}). "
                "Check the local embedding model and the uploaded PDF."
            )
            with st.chat_message("assistant"):
                st.error(answer)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "activity": []}
            )
            return

    if uploaded_pdfs and not pdf_context:
        st.info(
            "The vector search did not return PDF chunks for this question. "
            "The agent will be instructed not to guess document-specific details."
        )

    conversation = []
    for item in st.session_state.chat_history[:-1]:
        if item["role"] == "user":
            conversation.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            conversation.append(AIMessage(content=item["content"]))
    conversation.append(HumanMessage(content=question))

    with st.chat_message("assistant"):
        with st.spinner("Thinking and gathering relevant information..."):
            try:
                model = ChatOpenRouter(
                    model=model_name.strip(),
                    api_key=openrouter_api_key.strip(),
                    temperature=0,
                )
                activity = []
                document_answer = None
                if uploaded_pdfs and pdf_context:
                    with st.spinner("Checking whether the PDF answers your question..."):
                        document_answer = assess_document_support(
                            model,
                            conversation,
                            pdf_context,
                        )

                if document_answer and document_answer.sufficient:
                    answer = document_answer.answer.strip()
                    response_text = answer
                    st.markdown(answer)
                    st.caption(
                        "Answered from the uploaded PDF. No external search tools were called."
                    )
                    render_pdf_evidence(pdf_context)
                    with st.expander("Agent execution", expanded=False):
                        st.caption(
                            "Document support check: sufficient. Search tools were not invoked."
                        )
                else:
                    if document_answer:
                        st.caption(
                            "The PDF excerpts were checked first but did not fully answer "
                            "the question. The research agent may now use search tools."
                        )
                    agent = create_agent(
                        model,
                        create_search_tools(tavily_api_key.strip()),
                        system_prompt=get_system_prompt(
                            pdf_context,
                            pdf_was_truncated,
                            pdf_was_uploaded=bool(uploaded_pdfs),
                        ),
                        middleware=[
                            ToolCallLimitMiddleware(
                                run_limit=MAX_TOOL_CALLS,
                                exit_behavior="end",
                            )
                        ],
                    )
                    result = agent.invoke({"messages": conversation})
                    activity = collect_activity(result.get("messages", []))
                    final_message = next(
                        (
                            message
                            for message in reversed(result.get("messages", []))
                            if isinstance(message, AIMessage)
                            and not message.tool_calls
                            and message_text(message.content).strip()
                        ),
                        None,
                    )
                    answer = (
                        message_text(final_message.content).strip()
                        if final_message
                        else "The agent completed without returning a text answer."
                    )
                    response_text = answer
                    sources = collect_sources(activity)
                    if sources:
                        answer += "\n\n**Sources**\n" + "\n".join(
                            f"- [{title}]({url})" for title, url in sources
                        )
                    st.markdown(answer)
                    render_activity(activity)
                    render_pdf_evidence(pdf_context)
            except Exception as exc:
                answer = (
                    f"The agent request failed ({type(exc).__name__}). "
                    "Check the OpenRouter key, model name, and tool-calling support, then try again."
                )
                activity = []
                response_text = ""
                st.error(answer)

    assistant_turn = {
        "role": "assistant",
        "content": answer,
        "response": response_text,
        "question": question,
        "activity": activity,
        "pdf_context": pdf_context,
        "contexts": pdf_contexts + collect_contexts(activity, ""),
        "evaluation": None,
        "reference": "",
        "embedding_model": embedding_model,
    }
    st.session_state.chat_history.append(assistant_turn)
    render_ragas_evaluation(
        assistant_turn,
        len(st.session_state.chat_history) - 1,
        openrouter_api_key,
        model_name,
        embedding_model,
    )


if __name__ == "__main__":
    main()
