import copy
import json
import streamlit as st

from chat_dsat_mutator_controller import get_differences, mutate_chat_samples, generate_responses


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
                st.session_state.mutated_chat_samples, st.session_state.mutation_messages = mutate_chat_samples(st.session_state.model, copy.deepcopy(st.session_state.chat_samples), st.session_state.mutation_request, modified_mutation_messages)
                st.session_state.differences = get_differences(st.session_state.chat_samples, st.session_state.mutated_chat_samples)
                st.session_state.original_responses, st.session_state.new_responses = generate_responses(st.session_state.model, st.session_state.system_prompt, st.session_state.mutated_chat_samples)
            except Exception as e:
                st.error(e)
