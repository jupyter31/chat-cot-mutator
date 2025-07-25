import json
import streamlit as st


# define button functionality
def click_prev():
    if st.session_state.chat_index > 0:
        st.session_state.chat_index -= 1


def click_next():
    if st.session_state.chat_index < len(st.session_state.mutated_chat_samples) - 1:
        st.session_state.chat_index += 1


def download_all():
    st.download_button(
        label="Download ALL mutated chat samples (.jsonl)",
        data="\n".join([json.dumps(chat) for chat in st.session_state.mutated_chat_samples]),
        file_name="mutated_chat_samples.jsonl",
        mime="application/jsonl"
    )

    st.divider()


def display_individual_chat_sample_results():
    prev_chat, curr_chat, next_chat = st.columns([2.6,4,1])

    with prev_chat:
        st.button("⬅ Previous", key="prev_chat", on_click=click_prev)

    with curr_chat:
        st.subheader(f"Chat sample {st.session_state.chat_index + 1} of {len(st.session_state.mutated_chat_samples)}")

    with next_chat:
        st.button("Next ➡", key="next_chat", on_click=click_next)

    # define tabs for displaying results of the mutation
    tab1, tab2, tab3 = st.tabs(["Mutated chat sample", "Differences", "Responses"])

    # add download button for mutated chat sample, and display the mutated chat sample
    with tab1:
        st.write("")
        # TODO: make the data be the mutated chat sample + new response
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
        st.json(st.session_state.differences[st.session_state.chat_index])

    # display original response and new response
    with tab3:
        st.markdown("#### New response")
        st.write(st.session_state.new_responses[st.session_state.chat_index])

        st.markdown("#### Original response")
        st.write(st.session_state.original_responses[st.session_state.chat_index])
