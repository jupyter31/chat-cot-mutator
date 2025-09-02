import streamlit as st

from chat_dsat_mutator_controller import call_foundry_client

def get_diff_urls():

    st.write("Paste your foundry token below to generate links to the Copilot Playground Diff Tool for each chat sample.")
    st.write("Read the tooltip for instructions on how to obtain the token.")

    foundry_token = st.text_input(
        "Foundry Token",
        value=None,
        type="password",
        help=(
            "- Go to the Copilot Playground Diff Tool\n"
            "- Navigate to developer tools (Ctrl+Shift+I)\n"
            "- Click on the Network tab\n"
            "- Click the 'Create a sharable link to this diff' button at the top left of the page\n"
            "- Copy the token for the Authorization request header, after the word 'Bearer'\n"
        ),
        icon=":material/key:"
    )

    if st.button("Generate Diff Tool URLs", disabled=(not foundry_token)):
        with st.spinner("Generating URLs..."):
            st.session_state.diff_urls = []

            try:
                raw_diff_urls = call_foundry_client(
                    foundry_token,
                    [chat for chat, mut_chat in zip(st.session_state.chat_samples, st.session_state.mutated_chat_samples) if mut_chat],
                    [mut_chat for mut_chat in st.session_state.mutated_chat_samples if mut_chat]
                )

                st.session_state.diff_urls = [raw_diff_urls.pop(0) if mut_chat else None for mut_chat in st.session_state.mutated_chat_samples]

                st.session_state.show_diff_urls = True

            except Exception as e:
                st.error(f"Error generating URLs: {e}")
                st.session_state.diff_urls = []
                return

    if st.session_state.show_diff_urls:

        st.download_button(
            label="Download all Diff Tool URLs (.txt)",
            data="\n".join([url if url else "null" for url in st.session_state.diff_urls]),
            file_name="diff_urls.txt",
            type="tertiary",
            icon="⬇️"
        )

    st.divider()