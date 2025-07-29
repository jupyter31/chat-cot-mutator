import copy
import json
import streamlit as st

from chat_dsat_mutator_controller import get_differences, mutate_chat_samples, generate_responses


def click_prev():
    if st.session_state.msgs_index > 0:
        st.session_state.msgs_index -= 1


def click_next():
    if st.session_state.msgs_index < len(st.session_state.mutation_messages) - 1:
        st.session_state.msgs_index += 1


def edit_mutation_messages():
    for i, msgs in enumerate(st.session_state.mutation_messages):
        key = f"msgs_{i}"
        if key not in st.session_state:
            st.session_state[key] = json.dumps(msgs, indent=2)

    with st.expander("Edit messages", expanded=True):

        # define buttons for navigating through the individual chat samples
        prev_msgs, curr_msgs, next_msgs = st.columns([2.6,4,1])

        with prev_msgs:
            st.button("⬅ Previous", key="prev_msgs", on_click=click_prev)

        with curr_msgs:
            st.subheader(f"Chat sample {st.session_state.msgs_index + 1} of {len(st.session_state.mutation_messages)}")

        with next_msgs:
            st.button("Next ➡", key="next_msgs", on_click=click_next)

        st.session_state[f"msgs_{st.session_state.msgs_index}"] = st.text_area(
            "Messages",
            value=st.session_state[f"msgs_{st.session_state.msgs_index}"],
            height=400,
            disabled=False,
            label_visibility="collapsed"
        )

        # validate the new messages
        try:
            modified_mutation_messages = [json.loads(st.session_state[f"msgs_{i}"]) for i in range(len(st.session_state.mutation_messages))]
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
