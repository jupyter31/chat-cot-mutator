import json
import streamlit as st

from chat_mutator_controller import run_apology_judge, run_claimbreak, run_score_all
from mutation_data import Mutation

def run_hallucination_judge():
    st.write("Running the hallucination judge will evaluate the grounding of the new response.")
    st.write(
        "This will produce a score for the response from 0-1 which will give an indiciation of whether the response contains a hallucination. "
        "A score of 0 implies the the claims in the response are not present or clearly inferred from any search result. "
        "A score of 1 implies that the claims in the response appear verbatim in at least one search result, or can be inferred by combining data from the search results."
    )

    # get reasoning model to use
    st.markdown("##### Reasoning model")
    st.write("The reasoning model will be used evaluate the grounding of the new response.")
    st.session_state.reasoning_model = st.text_input(
        "To find more models to use, visit the [LLM API model list](https://substrate.microsoft.net/v2/llmApi/modelList)", 
        value=st.session_state.reasoning_model
    )

    if st.button("Run hallucination judge", disabled=(not st.session_state.reasoning_model)):
        if st.session_state.mutation_request in [Mutation.CLAIM_ALIGNED_DELETION, Mutation.SALIENCE_DROP]:
            with st.spinner("Identifying apologies in responses..."):
                try:
                    st.session_state.apology_classifications = run_apology_judge(st.session_state.reasoning_model, [mut_chat for mut_chat in st.session_state.mutated_chat_samples if mut_chat])
                except Exception as e:
                    st.error(f"Error running apology classifier: {e}")

        with st.spinner("Breaking responses into claims..."):
            try:
                st.session_state.claims = run_claimbreak(st.session_state.reasoning_model, [mut_chat for mut_chat in st.session_state.mutated_chat_samples if mut_chat])
                st.session_state.claims = [claims if (claims and claims != "[]") else None for claims in st.session_state.claims]
            except Exception as e:
                st.error(f"Error running claimbreak: {e}")

        with st.spinner("Scoring claims..."):
            try:
                st.session_state.reasonings, st.session_state.mean_scores = run_score_all(st.session_state.reasoning_model, [mut_chat for mut_chat in st.session_state.mutated_chat_samples if mut_chat], [claims for claims in st.session_state.claims if claims])

                st.session_state.apology_classifications = [st.session_state.apology_classifications.pop(0) if mut_chat else None for mut_chat in st.session_state.mutated_chat_samples]
                st.session_state.claims = [st.session_state.claims.pop(0) if mut_chat else None for mut_chat in st.session_state.mutated_chat_samples]
                st.session_state.reasonings = [st.session_state.reasonings.pop(0) if (mut_chat and claims) else None for mut_chat, claims in zip(st.session_state.mutated_chat_samples, st.session_state.claims)]
                st.session_state.mean_scores = [st.session_state.mean_scores.pop(0) if (mut_chat and claims) else None for mut_chat, claims in zip(st.session_state.mutated_chat_samples, st.session_state.claims)]

                st.session_state.show_scores = True

            except Exception as e:
                st.error(f"Error determining scores: {e}")

    if st.session_state.show_scores:

        st.download_button(
            label="Download individual apology classification reasoining (.txt)",
            data="\n".join(["null" if classification is None else json.dumps(classification).get("Reasoning") for classification in st.session_state.apology_classifications]),
            file_name="apology_classification_reasoning.jsonl",
            type="tertiary",
            icon="⬇️"
        )

        st.download_button(
            label="Download individual apology classification label (.txt)",
            data="\n".join(["null" if classification is None else json.dumps(classification).get("Label") for classification in st.session_state.apology_classifications]),
            file_name="apology_classification_labels.jsonl",
            type="tertiary",
            icon="⬇️"
        )

        st.download_button(
            label="Download individual claim score reasoning (.jsonl)",
            data="\n".join(["null" if reasoning is None else json.dumps(reasoning) for reasoning in st.session_state.reasonings]),
            file_name="claim_score_reasoning.jsonl",
            type="tertiary",
            icon="⬇️"
        ) 

        st.download_button(
            label="Download the mean score for all chat samples (.txt)",
            data="\n".join(["null" if mean_score is None else str(mean_score) for mean_score in st.session_state.mean_scores]),
            file_name="mean_chat_scores.txt",
            type="tertiary",
            icon="⬇️"
        ) 

    st.divider()