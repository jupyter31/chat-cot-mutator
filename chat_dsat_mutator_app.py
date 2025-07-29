import copy
import json
import streamlit as st

from chat_dsat_mutator_controller import get_differences, mutate_chat_samples, generate_responses

from components.system_prompt import edit_system_prompt, init_system_prompt
from components.mutation_messages import edit_mutation_messages
from components.results import display_individual_chat_sample_results, download_all


st.set_page_config(layout="centered", page_title="Chat DSAT Mutator", page_icon=":robot_face:")

# initialise session state with default values
def init_session_state(default_states):
    for state, default in default_states.items():
        if state not in st.session_state:
            st.session_state[state] = default

init_session_state({
    "chat_index": 0,
    "chat_samples": None,
    "differences": None,
    "model": "dev-gpt-4o-gg",
    "mutated_chat_samples": None,
    "mutation_messages": None,
    "mutation_request": None,
    "new_responses": None,
    "original_responses": None,
    "slider_params": {},
    "submit_click": False,
    "system_prompt": {}
})

init_system_prompt()

st.header("Synthetic Chat-Data Mutation Framework")

# get chat sample input from file upload or text area
st.subheader("Chat samples")
st.write(":blue-background[Please ensure that input chat samples are in a valid JSONL format, with each line being a valid JSON object.]")

uploaded_file = st.file_uploader("Upload a JSONL file of chat samples", type=["jsonl"])
raw_chat_samples = uploaded_file.read().decode("utf-8").strip().split("\n") if (uploaded_file is not None) else st.text_area("Paste chat samples here", height=170).strip().split("\n")

# validate chat samples
valid_chat_samples = True
if raw_chat_samples != ['']:
    try:
        st.session_state.chat_samples = [json.loads(chat) for chat in raw_chat_samples]
    except json.JSONDecodeError as e:
        valid_chat_samples = False
        st.error(f"Invalid JSON format: {e}")

# get mutation request
st.subheader("Mutation request")
mutation_options = ["Salience removal", "Claim-aligned deletion", "Topic dilution", "Negated-evidence injection", "Date / number jitter", "Passage shuffle", "Entity swap", "Document-snippet cut-off", "Unit-conversion rewrite", "Ablate URL links"]
mutation_request_selectbox = st.selectbox("Select mutation type", mutation_options, accept_new_options=False, index=None)
st.session_state.mutation_request = mutation_request_selectbox if mutation_request_selectbox is not None else st.text_input("Write your own mutation request", placeholder="e.g. 'Rewrite the chat sample with the dates swapped out for different dates.'").strip()


# get model to use
st.subheader("Model")
st.session_state.model = st.text_input(
    "To find more models to use, visit the [LLM API model list](https://substrate.microsoft.net/v2/llmApi/modelList)", 
    value="dev-gpt-4o-gg"
)

# enabled submit button if inputs are valid and a mutation request has been provided
disable_submit_button = (not valid_chat_samples) or (st.session_state.mutation_request.strip() == "") or (st.session_state.model.strip() == "")
submit = st.button("Submit", disabled=disable_submit_button)

st.divider()

# call LLM API Client when submit button is clicked
if submit:
    with st.spinner("Mutating chat samples..."):
        try:
            st.session_state.mutated_chat_samples, st.session_state.mutation_messages = mutate_chat_samples(st.session_state.model, copy.deepcopy(st.session_state.chat_samples), st.session_state.mutation_request)
            st.session_state.differences = get_differences(copy.deepcopy(st.session_state.chat_samples), copy.deepcopy(st.session_state.mutated_chat_samples))
            st.session_state.original_responses, st.session_state.new_responses = generate_responses(st.session_state.model, st.session_state.system_prompt, st.session_state.mutated_chat_samples)
            st.session_state.submit_click = True
        except Exception as e:
            st.error(e)

if st.session_state.submit_click:
    # TODO: only allow the change of the additional mutation messages, not original chat samples
    # show the messages used to mutate the chat samples and allow it to be modified and resubmitted
    st.subheader("Mutation messages")
    st.write("The messages below were used to produce the mutations. You can use it to understand how the mutations were generated, or modify the messages and regenerate the mutations.")

    edit_mutation_messages()

    # expose the system prompt used for generating the new responses
    st.subheader("System prompt")
    st.write("The parameters below were used in the system prompt to generate the new responses. You can use it to understand how the responses were generated, or modify the parameters and regenerate the responses.")
    
    edit_system_prompt()

    # TODO: make the data be the mutated chat samples + new responses
    # add download button for all mutated chat samples
    st.subheader("Mutated chat samples")
    
    download_all()

    st.subheader("Individual chat sample results")
    # TODO: make display load lighter
    # define buttons for navigating through the individual chat samples

    display_individual_chat_sample_results()
    