import json
import streamlit as st
import time

from mutation_data import Mutation

from chat_mutator_controller import run_full_process
from app_components.diff_urls import get_diff_urls
from app_components.judge import run_hallucination_judge
from app_components.mutation_request import init_mutation_customisations, edit_mutation_messages, get_mutation_request
from app_components.responses import regenerate_responses
from app_components.results import display_individual_chat_sample_results, download_all
from app_components.system_prompt import edit_system_prompt, init_system_prompt


# initialise session state with default values
def init_session_state(default_states):
    for state, default in default_states.items():
        if state not in st.session_state:
            st.session_state[state] = default


st.set_page_config(layout="centered", page_title="Chat Mutator", page_icon=":nerd_face:")
           
init_session_state({
    "chat_index": 0,
    "chat_model": "dev-gpt-5-chat-jj",
    "chat_samples": None,
    "customisations": {},
    "differences": None,
    "diff_urls": None,
    "errors": {},
    "key_suffix": 0,
    "mutated_chat_samples": None,
    "mutation_messages": None,
    "mutation_request": None,
    "new_responses": None,
    "original_responses": None,
    "reasoning_model": "dev-gpt-5-reasoning",
    "slider_params": {},
    "show_diff_urls": False,
    "show_results": False,
    "show_scores": False,
    "system_prompt": {}
})

init_mutation_customisations()

init_system_prompt()

st.header("Synthetic Chat-Data Mutation Framework")

# get chat sample input from file upload or text area
st.subheader("Chat samples")
st.write(":blue-background[Please ensure that input chat samples are in a valid JSONL format, with each line being a valid JSON object.]")

uploaded_file = st.file_uploader("Upload a JSONL file of chat samples", type=["jsonl"], on_change=lambda: st.session_state.update({"chat_index": 0}))
raw_chat_samples = uploaded_file.read().decode("utf-8").strip().split("\n") if (uploaded_file is not None) else st.text_area("Paste chat samples here", height=170).strip().split("\n")

# validate chat samples
valid_chat_samples = False

if raw_chat_samples != ['']:
    try:
        st.session_state.chat_samples = [json.loads(chat) for chat in raw_chat_samples]
        st.session_state.original_responses = [chat["messages"][-1]["content"] if chat["messages"][-1]["role"] == "assistant" else None for chat in st.session_state.chat_samples]
        valid_chat_samples = True
    except Exception as e:
        st.error(f"Invalid JSON format: {e}")

# get mutation request
st.subheader("Mutation request")

get_mutation_request()

valid_mutation_messages = False
if st.session_state.mutation_request != "":
    if (st.session_state.mutation_request == Mutation.PASSAGE_SHUFFLE and st.session_state.customisations.get("shuffle_depth") == "outer"):
        valid_mutation_messages = True
    else:
        # show the messages used to mutate the chat samples and allow it to be modified and resubmitted
        st.markdown("##### Mutation messages")
        st.write("The messages below were used to produce the mutations. You can use it to understand how the mutations were generated, or modify the messages and regenerate the mutations.")

        valid_mutation_messages = edit_mutation_messages()

# get chat model to use
st.subheader("Chat model")
st.write("The chat model will be used to perform the mutations and generate the new responses.")
st.session_state.chat_model = st.text_input(
    "To find more models to use, visit the [LLM API model list](https://substrate.microsoft.net/v2/llmApi/modelList)", 
    value=st.session_state.chat_model
)

# expose the system prompt used for generating the new responses
st.subheader("System prompt for response generation")
st.write("The parameters below were used in the system prompt to generate the new responses. You can use it to understand how the responses were generated, or modify the parameters and regenerate the responses.")
    
valid_system_prompt = edit_system_prompt()

# enabled submit button if inputs are valid and a mutation request has been provided
disable_submit_button = (not valid_chat_samples) or (st.session_state.mutation_request == "") or (not valid_mutation_messages) or (st.session_state.chat_model.strip() == "") or (not valid_system_prompt)
submit = st.button("Submit", disabled=disable_submit_button, type="primary")

st.divider()
if submit:
    st.session_state.start = time.time()
    with st.spinner("Mutating chat samples..."):
        try:
            st.session_state.chat_index = 0
            st.session_state.show_results = False
            st.session_state.show_diff_urls = False
            st.session_state.show_scores = False

            (
                st.session_state.mutated_chat_samples, 
                st.session_state.mutation_messages, 
                st.session_state.differences, 
                st.session_state.new_responses, 
                st.session_state.errors
            ) = run_full_process(st.session_state.chat_model, st.session_state.chat_samples, st.session_state.mutation_request, st.session_state.customisations, st.session_state.system_prompt, st.session_state.mutation_messages)

            st.session_state.end = time.time()
            print(f"Elapsed time: {st.session_state.end - st.session_state.start} seconds")
            
            st.session_state.show_results = True
        except Exception as e:
            st.error(e)

if st.session_state.show_results:

    # regenerate the assistant responses for the same mutated context
    st.subheader("Regenerate responses")
    regenerate_responses()

    # add download button for all mutated chat samples
    st.subheader("Mutated chat samples")
    download_all()

    # generate and display URLs for Copilot Playground Diff Tool
    st.subheader("Generate Diff Tool URLs")
    get_diff_urls()

    # select reasoning model and run hallucination judge
    st.subheader("Run hallucination judge")
    run_hallucination_judge()

    # define buttons for navigating through the individual chat samples
    st.subheader("Individual chat sample results")

    display_individual_chat_sample_results()
    