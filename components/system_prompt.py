import json
import streamlit as st

from chat_dsat_mutator_controller import add_new_responses_to_mutated_chat_samples, generate_responses

DEFAULT_SYSTEM_PROMPT_FILE = "system_prompts\\enterprise_copilot.json"

DEFAULT_SLIDER_PARAMS = {
    "temperature": {"label": "Temperature", "min": 0.0, "max": 2.0, "step": 0.1},
    "max_tokens": {"label": "Max Tokens", "min": 1, "max": 131072, "step": 1},
    "top_p": {"label": "Top P", "min": 0.0, "max": 1.0, "step": 0.1},
    "frequency_penalty": {"label": "Frequency Penalty", "min": -2.0, "max": 2.0, "step": 0.1},
    "presence_penalty": {"label": "Presence Penalty", "min": -2.0, "max": 2.0, "step": 0.1}
}


def init_system_prompt():
    if not st.session_state.system_prompt:
        with open(DEFAULT_SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            st.session_state.system_prompt = json.load(f)

    if not st.session_state.slider_params:
        st.session_state.slider_params = DEFAULT_SLIDER_PARAMS


def edit_system_prompt():
    with st.expander("Edit system prompt", expanded=True):

        # reset system prompt params to default values
        if st.button("Reset to default"):
            with open(DEFAULT_SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
                st.session_state.system_prompt = json.load(f)
                st.session_state.param_key_prefix += 1

        # allow numerical system prompt parameters to be modified using sliders
        for k, v in st.session_state.slider_params.items():
            st.markdown(f"**{v["label"]}**")
            st.session_state.system_prompt[k] = st.slider(
                v["label"],
                min_value=v["min"],
                max_value=v["max"],
                value=st.session_state.system_prompt[k],
                step=v["step"],
                key=f"{k}_{st.session_state.param_key_prefix}",
                label_visibility="collapsed"
            )

        # allow list of stop sequences to be modified
        st.markdown("**Stop Sequences**")
        st.session_state.system_prompt["stop"] = st.text_area(
            "Stop Sequences",
            value=("\n").join(st.session_state.system_prompt["stop"]),
            placeholder="e.g. '<|im_end|>'",
            key=f"stop_{st.session_state.param_key_prefix}",
            label_visibility="collapsed"
        ).strip().split("\n")
        
        # allow persona instructions to be modified
        st.markdown("**Messages**")
        modified_system_prompt_messages = st.text_area(
            "Messages",
            value=json.dumps(st.session_state.system_prompt["messages"], indent=2),
            height=400,
            disabled=False,
            key=f"messages_{st.session_state.param_key_prefix}",
            label_visibility="collapsed"
        )

        # validate persona instructions
        try:
            st.session_state.system_prompt["messages"] = json.loads(modified_system_prompt_messages.strip())
            valid_system_prompt = True
        except json.JSONDecodeError as e:
            valid_system_prompt = False
            st.error(f"Invalid JSON format in system prompt messages: {e}")

        disable_system_prompt_button = (not valid_system_prompt)
        regenerate = st.button("Regenerate responses with modified system prompt", disabled=disable_system_prompt_button)

    st.divider()
    
    if regenerate:
        with st.spinner("Regenerating responses..."):
            try:
                st.session_state.new_responses = generate_responses(st.session_state.model, st.session_state.system_prompt, st.session_state.mutated_chat_samples)
                st.session_state.mutated_chat_samples = add_new_responses_to_mutated_chat_samples(st.session_state.mutated_chat_samples, st.session_state.new_responses)
            except Exception as e:
                st.error(e)
