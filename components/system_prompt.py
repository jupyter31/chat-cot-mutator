import json
import streamlit as st

from chat_dsat_mutator_controller import generate_responses

def init_system_prompt():
    if not st.session_state.system_prompt:
        with open("system_prompts\\enterprise_copilot.json", "r", encoding="utf-8") as f:
            st.session_state.system_prompt = json.load(f)

    if not st.session_state.slider_params:
        st.session_state.slider_params = {
            "temperature": {"label": "Temperature", "min": 0.0, "max": 2.0, "step": 0.1},
            "max_tokens": {"label": "Max Tokens", "min": 1, "max": 131072, "step": 1},
            "top_p": {"label": "Top P", "min": 0.0, "max": 1.0, "step": 0.1},
            "frequency_penalty": {"label": "Frequency Penalty", "min": -2.0, "max": 2.0, "step": 0.1},
            "presence_penalty": {"label": "Presence Penalty", "min": -2.0, "max": 2.0, "step": 0.1}
        }


def edit_system_prompt():
    with st.expander("Edit system prompt", expanded=True):
        for k, v in st.session_state.slider_params.items():
            st.markdown(f"**{v["label"]}**")
            st.session_state.system_prompt[k] = st.slider(
                v["label"],
                min_value=v["min"],
                max_value=v["max"],
                value=st.session_state.system_prompt[k],
                step=v["step"],
                key=k,
                label_visibility="collapsed"
            )

        st.markdown("**Stop Sequences**")
        st.session_state.system_prompt["stop"] = st.text_area(
            "Stop Sequences",
            value=("\n").join(st.session_state.system_prompt["stop"]),
            placeholder="e.g. '<|im_end|>'",
            label_visibility="collapsed"
        ).strip().split("\n")
        
        st.markdown("**Messages**")
        modified_system_prompt_messages = st.text_area(
            "Messages",
            value=json.dumps(st.session_state.system_prompt["messages"], indent=2),
            height=400,
            disabled=False,
            label_visibility="collapsed"
        )

        try:
            st.session_state.system_prompt["messages"] = json.loads(modified_system_prompt_messages.strip())
            valid_system_prompt = True
        except json.JSONDecodeError as e:
            valid_system_prompt = False
            st.error(f"Invalid JSON format in system prompt messages: {e}")

        disable_system_prompt_button = (not valid_system_prompt)
        regen = st.button("Regenerate responses with modified system prompt", disabled=disable_system_prompt_button)

    st.divider()
    
    if regen:
        with st.spinner("Regenerating responses..."):
            try:
                st.session_state.original_responses, st.session_state.new_responses = generate_responses(st.session_state.model, st.session_state.system_prompt, st.session_state.mutated_chat_samples)
            except Exception as e:
                st.error(e)
