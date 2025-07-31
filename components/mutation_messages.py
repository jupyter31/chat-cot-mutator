import json
import streamlit as st

from chat_dsat_mutator_controller import run_full_process


def edit_mutation_messages():
    with st.expander("Edit messages", expanded=True):

        modified_mutation_messages = st.text_area(
            "Messages",
            value=json.dumps(st.session_state.mutation_messages, indent=2),
            height="content",
            disabled=False,
            label_visibility="collapsed"
        )

        # validate the new messages
        try:
            modified_mutation_messages = json.loads(modified_mutation_messages)
            valid_mutation_messages = True
        except json.JSONDecodeError as e:
            valid_mutation_messages = False
            st.error(f"Invalid JSON format in mutation messages: {e}")

        disable_retry_button = (not valid_mutation_messages)
        retry = st.button("Regenerate mutations with modified mutation messages", disabled=disable_retry_button)

    st.divider()

    if retry:        
        with st.spinner("Mutating chat samples..."):
            try:
                (st.session_state.mutated_chat_samples, st.session_state.mutation_messages, st.session_state.differences, st.session_state.new_responses) = run_full_process(st.session_state.model, st.session_state.chat_samples, st.session_state.mutation_request, st.session_state.system_prompt, modified_mutation_messages)   
            except Exception as e:
                st.error(e)
