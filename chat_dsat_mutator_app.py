import streamlit as st
import json
import difflib

from chat_dsat_mutator_controller import mutate_chat_sample

def format_diff_text(diff):
    """
    Formats the differences in a visual way for the Streamlit app.
    
    Args:
        diff (list): List of differences between the original and mutated chat sample.
    
    Returns:
        formatted_diff (str): The difference text with added visual formatting features.
    """

    formatted_diff = ""
    for word in diff:
        if word.startswith("+ "):
            formatted_diff += f":green-background[{word[2:]}] "
        elif word.startswith("- "):
            formatted_diff += f":red-background[~~{word[2:]}~~] "
        else:
            formatted_diff += f"{word} "

    formatted_diff = formatted_diff.strip()

    return formatted_diff

def show_mutation_diffs(original, mutations):
    """
    Displays the differences between the original chat sample and the mutated samples.
    
    Args:
        original (str): The original chat sample.
        mutations (list): List of mutated chat samples.
    """

    differences = [difflib.ndiff(original.split(), mut.split()) for mut in mutations]
    formatted_differences = [format_diff_text(diff) for diff in differences]

def validate_json(chat_sample):
    """
    Validates if the provided chat sample is a valid JSON.
    
    Args:
        chat_sample (str): The chat sample in JSON format.
        
    Returns:
        bool: True if valid, False otherwise.
    """
    if chat_sample is not None and chat_sample.strip() != "":
        try:
            chat_sample = json.loads(chat_sample)
            return True
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {e}")
            return False


st.header("Synthetic Chat-Data Mutation Framework")

# get chat sample input from file upload or text area
st.subheader("Chat sample")
uploaded_file = st.file_uploader("Upload a chat sample JSON file", type=["json"])

chat_sample = ""
valid_sample = False

if uploaded_file is not None:
    chat_sample = uploaded_file.read().decode("utf-8")
    valid_sample = validate_json(chat_sample)
else:
    chat_sample = st.text_area("Paste chat sample here")
    valid_sample = validate_json(chat_sample)

st.session_state["chat_sample"] = chat_sample

# get mutation request
st.subheader("Mutation request")

# TODO: Allow plain English mutation requests
#mutation_request = st.text_input("Enter mutation request", placeholder="e.g. 'Inject a hallucination'")

options = ["Misattribution", "Hallucination", "Policy edge-cases", "Persona shift"]
mutation_request = st.selectbox("Select mutation type", options, accept_new_options=False)

st.divider()

disable_button = (not valid_sample) or (chat_sample == "") or (mutation_request not in options)
mutations = st.button("Submit", on_click=mutate_chat_sample, args=(chat_sample, mutation_request), disabled=disable_button)



