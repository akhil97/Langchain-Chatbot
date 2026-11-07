import re
from typing import Any

import streamlit as st

from rag_services import evaluate_answer_with_deepeval, evaluate_answer_with_ragas


def render_activity(activity: list[dict[str, str]]) -> None:
    with st.expander("Agent execution", expanded=False):
        if not activity:
            st.caption("The agent answered without calling a search tool.")
            return
        for index, event in enumerate(activity, start=1):
            st.markdown(f"**{index}. `{event['tool']}`**")
            st.caption("Tool input")
            st.code(event["arguments"], language="json")
            st.caption("Retrieved result")
            linked_result = re.sub(
                r"(?m)^URL:\s*(https?://[^\s<>]+)",
                lambda match: f"URL: [{match.group(1)}]({match.group(1)})",
                event["result"],
            )
            st.markdown(linked_result)


def render_pdf_evidence(pdf_context: str) -> None:
    if pdf_context:
        with st.expander("PDF excerpts used for this answer", expanded=False):
            st.markdown(pdf_context)


def render_ragas_evaluation(
    item: dict[str, Any],
    turn_index: int,
    api_key: str,
    model_name: str,
    default_embedding_model: str,
) -> None:
    with st.expander("Evaluate this answer with RAGAS", expanded=False):
        if item.get("contexts"):
            st.caption(
                "Scores use this turn's actual retrieved context. Answer relevancy "
                "also uses OpenRouter's embeddings endpoint."
            )
        else:
            st.caption(
                "No retrieval context was used for this answer, so RAGAS will score "
                "answer relevancy only. Embeddings are requested through OpenRouter."
            )
        evaluation = item.get("evaluation")
        if evaluation:
            columns = st.columns(len(evaluation))
            for column, (metric, score) in zip(columns, evaluation.items()):
                column.metric(metric.replace("_", " ").title(), f"{score:.3f}")
            st.caption(
                "RAGAS scores range from 0 to 1. Context precision and recall are "
                "included only when you provide a reference answer and retrieved context exists."
            )

        with st.form(f"ragas-evaluation-{turn_index}"):
            reference = st.text_area(
                "Reference answer (optional; enables context precision and recall)",
                value=item.get("reference", ""),
                key=f"ragas-reference-{turn_index}",
                help="Enter a trusted expected answer to compare context coverage.",
            )
            embedding_model = st.text_input(
                "OpenRouter embedding model",
                value=item.get("embedding_model", default_embedding_model),
                key=f"ragas-embedding-model-{turn_index}",
            )
            submitted = st.form_submit_button("Run RAGAS evaluation")

        if submitted:
            if not item.get("response"):
                st.error("There is no successful answer to evaluate.")
            elif not api_key.strip():
                st.error("Add your OpenRouter API key in the sidebar before evaluating.")
            elif not model_name.strip() or not embedding_model.strip():
                st.error("Enter valid chat and embedding model names.")
            else:
                try:
                    with st.spinner("Scoring answer with RAGAS..."):
                        item["evaluation"] = evaluate_answer_with_ragas(
                            question=item["question"],
                            answer=item["response"],
                            contexts=item.get("contexts", []),
                            reference=reference,
                            api_key=api_key.strip(),
                            model_name=model_name.strip(),
                            embedding_model=embedding_model.strip(),
                        )
                    item["reference"] = reference
                    item["embedding_model"] = embedding_model
                    st.success("RAGAS evaluation complete.")
                    columns = st.columns(len(item["evaluation"]))
                    for column, (metric, score) in zip(
                        columns, item["evaluation"].items()
                    ):
                        column.metric(metric.replace("_", " ").title(), f"{score:.3f}")
                except Exception as exc:
                    st.error(
                        f"RAGAS evaluation failed ({type(exc).__name__}). "
                        "Check the OpenRouter model, API key, embedding access, "
                        "and reference/context inputs."
                    )


def render_deepeval_evaluation(
    item: dict[str, Any],
    turn_index: int,
    api_key: str,
    model_name: str,
) -> None:
    with st.expander("Evaluate this answer with DeepEval", expanded=False):
        st.caption(
            "Scores range from 0 to 1. Answer relevancy is always measured; "
            "faithfulness and contextual relevancy require retrieved context. "
            "Add a reference answer to include contextual precision and recall."
        )
        evaluation = item.get("deepeval_evaluation")
        if evaluation:
            columns = st.columns(len(evaluation))
            for column, (metric, result) in zip(columns, evaluation.items()):
                column.metric(
                    metric.replace("_", " ").title(),
                    f"{result['score']:.3f}",
                    "PASS" if result["passed"] else "BELOW 0.70",
                )
            with st.expander("Metric explanations"):
                for metric, result in evaluation.items():
                    st.markdown(
                        f"**{metric.replace('_', ' ').title()}** — "
                        f"{'Passed' if result['passed'] else 'Below threshold'}"
                    )
                    st.write(result["reason"] or "No explanation returned.")

        with st.form(f"deepeval-evaluation-{turn_index}"):
            reference = st.text_area(
                "Reference answer (optional)",
                value=item.get("deepeval_reference", ""),
                key=f"deepeval-reference-{turn_index}",
                help=(
                    "A trusted expected answer enables contextual precision and "
                    "contextual recall when retrieved context exists."
                ),
            )
            submitted = st.form_submit_button("Run DeepEval evaluation")

        if submitted:
            if not item.get("response"):
                st.error("There is no successful answer to evaluate.")
            elif not api_key.strip():
                st.error("Add your OpenRouter API key in the sidebar before evaluating.")
            elif not model_name.strip():
                st.error("Enter a valid OpenRouter model name.")
            else:
                try:
                    with st.spinner("Scoring answer with DeepEval..."):
                        item["deepeval_evaluation"] = evaluate_answer_with_deepeval(
                            question=item["question"],
                            answer=item["response"],
                            contexts=item.get("contexts", []),
                            reference=reference,
                            api_key=api_key.strip(),
                            model_name=model_name.strip(),
                        )
                    item["deepeval_reference"] = reference
                    st.success("DeepEval evaluation complete.")
                    columns = st.columns(len(item["deepeval_evaluation"]))
                    for column, (metric, result) in zip(
                        columns, item["deepeval_evaluation"].items()
                    ):
                        column.metric(
                            metric.replace("_", " ").title(),
                            f"{result['score']:.3f}",
                            "PASS" if result["passed"] else "BELOW 0.70",
                        )
                except Exception as exc:
                    st.error(
                        f"DeepEval evaluation failed ({type(exc).__name__}). "
                        "Check the OpenRouter key, model name, and model availability."
                    )
