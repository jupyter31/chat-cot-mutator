import streamlit as st
import json
from deepdiff import DeepDiff
from chat_dsat_mutator_controller import mutate_chat_sample


st.header("Synthetic Chat-Data Mutation Framework")

# get chat sample input from file upload or text area
st.subheader("Chat sample")
uploaded_file = st.file_uploader("Upload a chat sample JSON file", type=["json"])


str_chat_sample = uploaded_file.read().decode("utf-8") if (uploaded_file is not None) else st.text_area("Paste chat sample here", height=170)
json_chat_sample = None

if str_chat_sample.strip() != "":
    try:
        json_chat_sample = json.loads(str_chat_sample)
    except json.JSONDecodeError as e:
        json_chat_sample = None
        st.error(f"Invalid JSON format: {e}")

# get mutation request
st.subheader("Mutation request")

# TODO : get plain English mutation requests
# mutation_request = ""
# mutation_request = st.text_input("Enter mutation request", placeholder="e.g. 'Perform entity swapping on the chat sample'")
# disable_button = (not valid_sample) or (chat_sample == "") or (mutation_request.strip() == "")

options = ["Salience removal", "Claim-aligned deletion", "Topic dilution", "Negated-evidence injection", "Date / number jitter", "Passage shuffle", "Entity swap", "Document-snippet cut-off", "Unit-conversion rewrite", "Ablate URL links"]
mutation_request = st.selectbox("Select mutation type", options, accept_new_options=False)
disable_button = (json_chat_sample is None) or (mutation_request.strip() == "")

# get the number of variants (different mutation options) requested
st.subheader("Number of variants")

st.number_input("Number of mutations", min_value=1, max_value=10, value=1, key="num_mutations")

submit_click = st.button("Submit", disabled=disable_button)

st.divider()

if submit_click:
    st.subheader("Mutated variants")
    mutations = mutate_chat_sample(str_chat_sample, mutation_request)

    for i, mut in enumerate(mutations):
        st.write(f"#### Variant {i + 1}")
        st.json(mut)

        # TODO: add diff highlighting
        # diff = DeepDiff(json_chat_sample, mut, view="text")

        st.download_button(
            label="Download JSON",
            data=json.dumps(mut, indent=2),
            file_name=f"mutated_variant_{i+1}.json",
            mime="application/json"
        )

