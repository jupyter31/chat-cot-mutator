import copy
import json
import streamlit as st

from chat_dsat_mutator_controller import get_differences, mutate_chat_samples, mutate_chat_samples_given_prompts, generate_responses

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
    "msgs_index": 0,
    "model": "dev-gpt-4o-gg",
    "mutated_chat_samples": None,
    "mutation_messages": None,
    "mutation_request": None,
    "new_responses": None,
    "original_responses": None,
    "submit_click": False,
    "system_prompt_params": {}
})

# define button functionality
def click_prev_chat():
    if st.session_state.chat_index > 0:
        st.session_state.chat_index -= 1

def click_next_chat():
    if st.session_state.chat_index < len(st.session_state.mutated_chat_samples) - 1:
        st.session_state.chat_index += 1

def click_prev_msgs():
    if st.session_state.msgs_index > 0:
        st.session_state.msgs_index -= 1

def click_next_msgs():
    if st.session_state.msgs_index < len(st.session_state.mutation_messages) - 1:
        st.session_state.msgs_index += 1


# set default input validity
valid_chat_samples = True
valid_mutation_messages = True


st.header("Synthetic Chat-Data Mutation Framework")

# get chat sample input from file upload or text area
st.subheader("Chat samples")
st.write(":blue-background[Please ensure that input chat samples are in a valid JSONL format, with each line being a valid JSON object.]")

uploaded_file = st.file_uploader("Upload a JSONL file of chat samples", type=["jsonl"])
raw_chat_samples = uploaded_file.read().decode("utf-8").strip().split("\n") if (uploaded_file is not None) else st.text_area("Paste chat samples here", height=170).strip().split("\n")

# validate chat samples
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
            st.session_state.original_responses, st.session_state.new_responses = generate_responses(st.session_state.model, st.session_state.mutated_chat_samples)
            st.session_state.submit_click = True
        except Exception as e:
            st.error(e)


# show the messages used to mutate the chat samples and allow it to be modified and resubmitted
if st.session_state.submit_click:
    st.subheader("Mutation messages")
    st.write("The messages below were used to produce the mutations. You can use it to understand how the mutations were generated, or modify the messages and regenerate the mutations.")

    for i, msgs in enumerate(st.session_state.mutation_messages):
        key = f"msgs_{i}"
        if key not in st.session_state:
            st.session_state[key] = json.dumps(msgs, indent=2)

    with st.expander("Edit messages", expanded=False):

        # define buttons for navigating through the individual chat samples
        prev_msgs, curr_msgs, next_msgs = st.columns([2.6,4,1])

        with prev_msgs:
            st.button("⬅ Previous", key="prev_msgs", on_click=click_prev_msgs)

        with curr_msgs:
            st.subheader(f"Chat sample {st.session_state.msgs_index + 1} of {len(st.session_state.mutation_messages)}")

        with next_msgs:
            st.button("Next ➡", key="next_msgs", on_click=click_next_msgs)

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
        except json.JSONDecodeError as e:
            valid_mutation_messages = False
            st.error(f"Invalid JSON format in mutation messages: {e}")

        disable_retry_button = (not valid_mutation_messages)
        retry = st.button("Regenerate mutations with modified mutation messages", disabled=disable_retry_button)

    st.divider()

    if retry:        
        with st.spinner("Mutating chat samples..."):
            try:
                st.session_state.mutated_chat_samples, st.session_state.mutation_messages = mutate_chat_samples_given_prompts(st.session_state.model, copy.deepcopy(st.session_state.chat_samples), modified_mutation_messages, st.session_state.mutation_request)
                st.session_state.differences = get_differences(st.session_state.chat_samples, st.session_state.mutated_chat_samples)
                st.session_state.original_responses, st.session_state.new_responses = generate_responses(st.session_state.model, st.session_state.mutated_chat_samples)
            except Exception as e:
                st.error(e)

    st.subheader("Mutated chat samples")

    # add download button for all mutated chat samples
    st.download_button(
        label="Download ALL mutated chat samples (.jsonl)",
        data="\n".join([json.dumps(chat) for chat in st.session_state.mutated_chat_samples]),
        file_name="mutated_chat_samples.jsonl",
        mime="application/jsonl"
    )

    st.divider()

    st.subheader("System prompt")
    st.write("The parameters below were used in the system prompt to generate the new responses. You can use it to understand how the responses were generated, or modify the parameters and regenerate the responses.")

    # TODO: move this somewhere else
    with open("persona_instructions\\enterprise_copilot_system_prompt.json", "r", encoding="utf-8") as f:
        persona_instructions = json.load(f)

    # TODO: change value to be taken from system prompt
    # TODO: simplify using list of dicts to generalise
    with st.expander("Edit system prompt", expanded=False):
        st.markdown("Number of Responses")
        st.session_state.system_prompt_params["n"] = st.slider("Number of Responses", min_value=1, max_value=10, value=1, step=1, label_visibility="collapsed")
        st.markdown("Temperature")
        st.session_state.system_prompt_params["temperature"] = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1, label_visibility="collapsed")
        st.markdown("Max Tokens")
        st.session_state.system_prompt_params["max_tokens"] = st.slider("Max Tokens", min_value=1, max_value=131072, value=4096, step=1, label_visibility="collapsed")
        st.markdown("Top P")
        st.session_state.system_prompt_params["top_p"] = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, step=0.1, label_visibility="collapsed")
        st.markdown("Frequency Penalty")
        st.session_state.system_prompt_params["frequency_penalty"] = st.slider("Frequency Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.1, label_visibility="collapsed")
        st.markdown("Presence Penalty")
        st.session_state.system_prompt_params["presence_penalty"] = st.slider("Presence Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.1, label_visibility="collapsed")
        st.markdown("Stop Sequences")
        st.session_state.system_prompt_params["stop"] = st.text_area("Stop Sequences", value="<|im_end|>\n<|im_start|>\n<|fim_suffix|>", placeholder="e.g. '<|im_end|>'", label_visibility="collapsed").strip().split("\n")
        st.markdown("Messages")
        st.session_state.system_prompt_params["messages"] = st.text_area(
            "Messages",
            value=json.dumps(persona_instructions["messages"], indent=2),
            height=400,
            disabled=False,
            label_visibility="collapsed"
        )

    st.divider()

    # define buttons for navigating through the individual chat samples
    prev_chat, curr_chat, next_chat = st.columns([2.6,4,1])

    with prev_chat:
        st.button("⬅ Previous", key="prev_chat", on_click=click_prev_chat)

    with curr_chat:
        st.subheader(f"Chat sample {st.session_state.chat_index + 1} of {len(st.session_state.mutated_chat_samples)}")

    with next_chat:
        st.button("Next ➡", key="next_chat", on_click=click_next_chat)

    # define tabs for displaying results of the mutation
    tab1, tab2, tab3 = st.tabs(["Mutated chat sample", "Differences", "Responses"])

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
        st.json(st.session_state.differences[st.session_state.chat_index])

    # display original response and new response
    with tab3:
        st.markdown("#### New response")
        st.write(st.session_state.new_responses[st.session_state.chat_index])

        st.markdown("#### Original response")
        st.write(st.session_state.original_responses[st.session_state.chat_index])
