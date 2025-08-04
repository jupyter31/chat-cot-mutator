import json
import streamlit as st

from mutation_data import get_mutation_messages


MUTATION_OPTIONS = ["Salience drop", "Claim-aligned deletion", "Topic dilution", "Negated-evidence injection", "Date / number jitter", "Passage shuffle", "Entity swap", "Document-snippet cut-off", "Unit-conversion rewrite", "Ablate URL links"]

def edit_mutation_messages():
    with st.expander("Edit messages", expanded=True):

        # reset mutation messages to default values
        if st.button("Reset to default", key="reset_mutation_messages"):
            st.session_state.mutation_messages = list(get_mutation_messages(st.session_state.mutation_request))
            st.session_state.param_key_prefix += 1


        modified_mutation_messages = st.text_area(
            "Messages",
            value=json.dumps(st.session_state.mutation_messages, indent=2),
            height="content",
            key=f"mutation_messages_{st.session_state.param_key_prefix}",
            disabled=False,
            label_visibility="collapsed"
        )

        # validate the new messages
        try:
            st.session_state.mutation_messages = json.loads(modified_mutation_messages)
            return True
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format in mutation messages: {e}")
            return False
        

def get_mutation_request():
    mutation_request_selectbox = st.selectbox("Select mutation type", MUTATION_OPTIONS, accept_new_options=False, index=None)
    st.session_state.mutation_request = mutation_request_selectbox if mutation_request_selectbox is not None else st.text_input("Write your own mutation request", placeholder="e.g. 'Rewrite the chat sample with the dates swapped out for different dates.'").strip()
    st.session_state.mutation_messages = list(get_mutation_messages(st.session_state.mutation_request))
