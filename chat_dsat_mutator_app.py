import copy
import json
import streamlit as st

from chat_dsat_mutator_controller import get_differences, mutate_chat_samples, mutate_chat_samples_given_prompts, regenerate_responses

st.set_page_config(layout="centered", page_title="Chat DSAT Mutator", page_icon=":robot_face:")

COMPONENT_WIDTH = 700

# initialise session state with default values
def init_session_state(default_states):
    for state, default in default_states.items():
        if state not in st.session_state:
            st.session_state[state] = default

def prev_chat():
    if st.session_state.chat_index > 0:
        st.session_state.chat_index -= 1


def next_chat():
    if st.session_state.chat_index < len(st.session_state.mutated_chat_samples) - 1:
        st.session_state.chat_index += 1

init_session_state({
    "chat_index": 0,
    "chat_samples": None,
    "differences": None,
    "mutated_chat_samples": None,
    "mutation_messages": None,
    "mutation_request": None,
    "new_responses": None,
    "original_responses": None,
    "retry_click": False,
    "submit_click": False,
})

# set default input 
valid_chat_samples = False
valid_mutation_messages = False


st.header("Synthetic Chat-Data Mutation Framework")

# get chat sample input from file upload or text area
st.subheader("Chat samples")
st.write(":blue-background[Please ensure that input chat samples are in a valid JSONL format, with each line being a valid JSON object.]")

uploaded_file = st.file_uploader("Upload a JSONL file of chat samples", type=["jsonl"], width=COMPONENT_WIDTH)
filename = uploaded_file.name.strip() if (uploaded_file is not None) else ""

raw_chat_samples = uploaded_file.read().decode("utf-8").strip().split("\n") if (uploaded_file is not None) else st.text_area("Paste chat samples here", height=170).strip().split("\n")

# validate chat samples
if raw_chat_samples != ['']:
    valid_chat_samples = True

    try:
        st.session_state.chat_samples = [json.loads(chat) for chat in raw_chat_samples]
    except json.JSONDecodeError as e:
        valid_chat_samples = False
        st.error(f"Invalid JSON format: {e}")


# get mutation request
st.subheader("Mutation request")

options = ["Salience removal", "Claim-aligned deletion", "Topic dilution", "Negated-evidence injection", "Date / number jitter", "Passage shuffle", "Entity swap", "Document-snippet cut-off", "Unit-conversion rewrite", "Ablate URL links"]
mutation_request_selectbox = st.selectbox("Select mutation type", options, accept_new_options=False, index=None, width=COMPONENT_WIDTH)
st.session_state.mutation_request = mutation_request_selectbox if mutation_request_selectbox is not None else st.text_input("Write your own mutation request", placeholder="e.g. 'Rewrite the chat sample with the dates swapped out for different dates.'", width=COMPONENT_WIDTH).strip()


# enabled submit button if inputs are valid and a mutation request has been provided
disable_submit_button = (not valid_chat_samples) or (st.session_state.mutation_request.strip() == "")
submit = st.button("Submit", disabled=disable_submit_button)


st.divider()


# call LLM API Client when submit button is clicked
if submit:
    with st.spinner("Mutating chat samples..."):
        st.session_state.mutated_chat_samples, st.session_state.mutation_messages = mutate_chat_samples(copy.deepcopy(st.session_state.chat_samples), st.session_state.mutation_request)
        st.session_state.differences = get_differences(st.session_state.chat_samples, st.session_state.mutated_chat_samples)
        st.session_state.original_responses, st.session_state.new_responses = regenerate_responses(st.session_state.mutated_chat_samples)

        st.session_state.submit_click = True
        st.session_state.retry_click = False


# show the prompt used to mutate the chat samples and allow it to be modified and resubmitted
if st.session_state.submit_click:
    st.subheader("Mutation messages")
    st.write("The messages below were used to produce the mutations. You can use it to understand how the mutations were generated, or modify and regenerate them.")

    # TODO : display this in a better way
    with st.expander("Messages", expanded=True):
        raw_modified_mutation_messages = st.text_area(
            "Messages",
            value="\n\n".join([json.dumps(msgs, indent=2) for msgs in st.session_state.mutation_messages]),
            height=300,
            disabled=False,
            label_visibility="hidden"
        )

        modified = (raw_modified_mutation_messages != "\n\n".join([json.dumps(msgs, indent=2) for msgs in st.session_state.mutation_messages]))

        # validate the new mutation_messages
        if modified:
            valid_mutation_messages = True
            try:
                modified_mutation_messages = [json.loads(msgs) for msgs in raw_modified_mutation_messages.strip().split("\n\n")]
            except json.JSONDecodeError as e:
                valid_mutation_messages = False
                st.error(f"Invalid JSON format in mutation messages: {e}")

        disable_retry_button = (not valid_mutation_messages) or (not modified)
        retry = st.button("Regenerate mutations with modified mutation messages", disabled=disable_retry_button)

    st.divider()

    if retry:        
        with st.spinner("Mutating chat samples..."):
            st.session_state.mutated_chat_samples, st.session_state.mutation_messages = mutate_chat_samples_given_prompts(copy.deepcopy(st.session_state.chat_samples), modified_mutation_messages, st.session_state.mutation_request)
            st.session_state.differences = get_differences(st.session_state.chat_samples, st.session_state.mutated_chat_samples)
            st.session_state.original_responses, st.session_state.new_responses = regenerate_responses(st.session_state.mutated_chat_samples)

            st.session_state.retry_click = True
            st.session_state.submit_click = False

if st.session_state.submit_click or st.session_state.retry_click:

    st.subheader("Mutated chat samples")

    # download button for all mutates chat samples
    st.download_button(
        label="Download ALL mutated chat samples (.jsonl)",
        data="\n".join([json.dumps(chat) for chat in st.session_state.mutated_chat_samples]),
        file_name=f"{filename}_mutated_samples.jsonl",
        mime="application/jsonl"
    )


    prev, curr, next = st.columns([1,2,1])

    with prev:
        st.button("⬅ Previous", on_click=prev_chat)

    with curr:
        st.subheader(f"Chat sample {st.session_state.chat_index + 1} of {len(st.session_state.mutated_chat_samples)}")

    with next:
        st.button("Next ➡", on_click=next_chat)

    tab1, tab2, tab3 = st.tabs(["Mutated chat sample", "Differences", "Responses"], width="stretch")

    with tab1:
        st.download_button(
            label=f"Download mutation of chat sample {st.session_state.chat_index + 1} (.json)",
            data=json.dumps(st.session_state.mutated_chat_samples[st.session_state.chat_index], indent=2),
            file_name=f"{filename}_mutated_chat_sample_{st.session_state.chat_index + 1}.json",
            mime="application/json"
        )
        st.json(st.session_state.mutated_chat_samples[st.session_state.chat_index])

    with tab2:
        st.json(st.session_state.differences[st.session_state.chat_index])

    with tab3:
        st.write(st.session_state.original_responses[st.session_state.chat_index])
        st.write(st.session_state.new_responses[st.session_state.chat_index])


