import streamlit as st

from chat_dsat_mutator_controller import add_new_responses_to_mutated_chat_samples, generate_responses, MAX_RETRY

def regenerate_responses():
    st.write("This will regenerate the assistant responses for the same mutated context.")
    st.write("This process will overwrite the existing responses, so please ensure you have downloaded any important data before proceeding.")

    if st.button("Regenerate responses"):
        with st.spinner("Generating new assistant responses..."):
            try:
                st.session_state.chat_index = 0
                st.session_state.show_diff_urls = False
                st.session_state.show_scores = False

                # generate new responses for all successfully mutated chat samples
                print("Generating new assistant responses...")
                successes = [i for i, mut_chat in enumerate(st.session_state.mutated_chat_samples) if mut_chat]
                raw_responses = generate_responses(st.session_state.chat_model, st.session_state.system_prompt, [st.session_state.mutated_chat_samples[i] for i in successes])
                res_successes = [i for i, response in zip(successes, raw_responses) if response is not None]

                # remove any response generation errors from the previous run
                st.session_state.errors = {i: err for i, err in st.session_state.errors.items() if not err.startswith("Response generation")}

                # add any new response generation errors
                st.session_state.errors.update({i: f"Response generation failed after {MAX_RETRY} attempts." for i, response in zip(successes, raw_responses) if response is None})

                # add the new assistant responses to the mutated contexts
                raw_mutated_chat_samples = add_new_responses_to_mutated_chat_samples([st.session_state.mutated_chat_samples[i] for i in res_successes], [raw_responses[successes.index(i)] for i in res_successes])
                
                for i in res_successes:
                    st.session_state.mutated_chat_samples[i] = raw_mutated_chat_samples[res_successes.index(i)]
                    st.session_state.new_responses[i] = raw_responses[successes.index(i)]
                
            except Exception as e:
                st.error(e)

    st.divider()