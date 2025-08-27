import streamlit as st

from chat_dsat_mutator_controller import run_claimbreak, run_score_all

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
        with st.spinner("Breaking responses into claims..."):
            try:
                claims = run_claimbreak(st.session_state.reasoning_model, [mut for mut in st.session_state.mutated_chat_samples if mut])
            except Exception as e:
                st.error(f"Error running claimbreak: {e}")

        with st.spinner("Scoring claims..."):
            try:
                scores = run_score_all(st.session_state.reasoning_model, [mut for mut in st.session_state.mutated_chat_samples if mut], claims)
            except Exception as e:
                st.error(f"Error determining scores: {e}")

    st.divider()