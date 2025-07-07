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



st.header("Synthetic Chat-Data Mutation Framework")

# get chat sample input from file upload or text area
st.subheader("Chat sample")
# uploaded_file = st.file_uploader("Upload a JSON file", type=["json"])
# chat_sample = ""

# if uploaded_file is not None:
#     try:
#         chat_sample = json.load(uploaded_file)
#         chat_sample = json.dumps(chat_sample, indent=2)
#     except Exception as e:
#         st.error(f"Error reading JSON file: {e}")
# else:
#     chat_sample = st.text_area("Paste chat sample here")

chat_sample = st.text_area("Paste chat sample here")

st.session_state["chat_sample"] = chat_sample

# get mutation request
st.subheader("Mutation request")
#mutation_request = st.text_input("Enter mutation request", placeholder="e.g. 'Inject a hallucination'")
options = ["Misattribution", "Hallucination", "Policy edge-cases", "Persona shift"]
mutation_request = st.selectbox("Select mutation type", options, accept_new_options=False)

mutations = mutate_chat_sample(chat_sample, mutation_request)


