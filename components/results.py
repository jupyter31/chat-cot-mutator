import json
import streamlit as st

from clients.foundry import foundry_client


def click_prev():
    if st.session_state.chat_index > 0:
        st.session_state.chat_index -= 1


def click_next():
    if st.session_state.chat_index < len(st.session_state.chat_samples) - 1:
        st.session_state.chat_index += 1


def download_all():
    st.download_button(
        label="Download ALL mutated chat samples (.jsonl)",
        data="\n".join([json.dumps(chat) for chat in st.session_state.mutated_chat_samples]),
        file_name="mutated_chat_samples.jsonl",
        mime="application/jsonl"
    )

    for k, v in st.session_state.errors.items():
        st.error(f"Chat sample {k + 1} failed with error: {v}")

    st.divider()


def display_individual_chat_sample_results():
    prev_chat, curr_chat, next_chat = st.columns([2.6,4,1])

    with prev_chat:
        st.button("⬅ Previous", key="prev_chat", on_click=click_prev)

    with curr_chat:
        st.subheader(f"Chat sample {st.session_state.chat_index + 1} of {len(st.session_state.chat_samples)}")

    with next_chat:
        st.button("Next ➡", key="next_chat", on_click=click_next)

    # define tabs for displaying results of the mutation
    tab1, tab2, tab3 = st.tabs(["Mutated chat sample", "Differences", "Responses"])

    if st.session_state.chat_index not in (st.session_state.errors).keys():
        # add download button for mutated chat sample, and display the mutated chat sample
        with tab1:
            st.write("")
            st.download_button(
                label=f"Download mutation of chat sample {st.session_state.chat_index + 1} (.json)",
                data=json.dumps(st.session_state.mutated_chat_samples[st.session_state.chat_index], indent=2),
                file_name=f"mutated_chat_sample_{st.session_state.chat_index + 1}.json",
                mime="application/json"
            )
            st.write("")

            st.json(st.session_state.mutated_chat_samples[st.session_state.chat_index])

        # show differences between original and mutated chat sample
        with tab2:
            diff_url = foundry_client.save_diff(
                json.dumps(st.session_state.chat_samples[st.session_state.chat_index], indent=2),
                json.dumps(st.session_state.mutated_chat_samples[st.session_state.chat_index], indent=2),
                "Original chat sample", 
                "Mutated chat sample",
            )

            st.markdown(f"[Click here to see the mutations using the Copilot Playground Diff Tool]({diff_url})")
            st.write("")
            st.json(st.session_state.differences[st.session_state.chat_index])

        # display original response and new response
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Original response")
                st.write(st.session_state.original_responses[st.session_state.chat_index])
            with col2:
                st.markdown("#### New response")
                st.write(st.session_state.new_responses[st.session_state.chat_index])

    else:
        st.error(f"Chat sample {st.session_state.chat_index + 1} failed with error: {st.session_state.errors[st.session_state.chat_index]}")        
