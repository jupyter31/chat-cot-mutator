import streamlit as st
import json
from chat_dsat_mutator_controller import foo



st.header("Synthetic Chat-Data Mutation Framework")

# get chat sample input from file upload or text area
st.subheader("Chat sample")
uploaded_file = st.file_uploader("Upload a JSON file", type=["json"])
chat_sample = ""

if uploaded_file is not None:
    try:
        chat_sample = json.load(uploaded_file)
        chat_sample = json.dumps(chat_sample, indent=2)
    except Exception as e:
        st.error(f"Error reading JSON file: {e}")
else:
    chat_sample = st.text_area("Paste chat sample here")

st.session_state["chat_sample"] = chat_sample

# get mutation request
st.subheader("Mutation request")
#mutation_request = st.text_input("Enter mutation request", placeholder="e.g. 'Inject a hallucination'")
options = ["Misattribution", "Hallucination", "Policy edge-cases", "Persona shift"]
mutation_request = st.selectbox("Select mutation type", options, accept_new_options=False)


