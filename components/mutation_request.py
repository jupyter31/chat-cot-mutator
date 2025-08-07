import json
import streamlit as st

from mutation_data import get_mutation_messages, Mutation, MUTATION_MAPPING, DEFAULT_MUTATION_CUSTOMISATIONS


def init_mutation_customisations():
    for mutation_type, params in DEFAULT_MUTATION_CUSTOMISATIONS.items():
        mut_str = mutation_type.name.lower()
        for param, default_value in params.items():
            if f"{mut_str}_{param}" not in st.session_state:
                st.session_state[f"{mut_str}_{param}"] = default_value


def edit_mutation_messages():
    with st.expander("Edit messages", expanded=False):

        modified_mutation_messages = st.text_area(
            "Messages",
            value=json.dumps(st.session_state.mutation_messages, indent=2),
            height="content",
            disabled=False,
            label_visibility="collapsed"
        )

        # validate the new messages
        try:
            st.session_state.mutation_messages = json.loads(modified_mutation_messages)
            return True
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format in mutation messages: {e}")
            return False
        

def get_mutation_request():
    mutation_request_selectbox = st.selectbox("Select mutation type", MUTATION_MAPPING.keys(), format_func=lambda x:MUTATION_MAPPING[x], accept_new_options=False, index=None)
    st.session_state.mutation_request = mutation_request_selectbox if mutation_request_selectbox is not None else st.text_input("Write your own mutation request", placeholder="e.g. 'Rewrite the chat sample with the dates swapped out for different dates.'").strip()

    # reset customisations and mutation messages to default values
    if st.button("Reset to default", key="reset_mutation_messages"):
        for mutation_type, params in DEFAULT_MUTATION_CUSTOMISATIONS.items():
            mut_str = mutation_type.name.lower()
            for param, default_value in params.items():
                st.session_state[f"{mut_str}_{param}"] = default_value

        st.session_state.mutation_messages = list(get_mutation_messages(st.session_state.mutation_request))

    customisations = get_mutation_customisation()
    st.session_state.mutation_messages = list(get_mutation_messages(st.session_state.mutation_request, customisations))


def get_mutation_customisation():
    if st.session_state.mutation_request in Mutation:
        st.markdown("##### Mutation customisations")
        mut_str = (st.session_state.mutation_request).name.lower()
        
    match st.session_state.mutation_request:

        case Mutation.SALIENCE_DROP:
            st.session_state[f"{mut_str}_number"] = st.session_state[f"{mut_str}_number"]
            number = st.slider(
                "Select the number of salient passages to drop",
                min_value=1,
                max_value=5,
                step=1,
                key=f"{mut_str}_number"
            )

            return {"number": number}
        
        case Mutation.CLAIM_ALIGNED_DELETION:
            # TODO
            pass

        case Mutation.TOPIC_DILUTION:
            st.session_state[f"{mut_str}_level"] = st.session_state[f"{mut_str}_level"]
            level = st.radio(
                "Select the plausibility of the topic dilution",
                options=["high", "medium", "low"],
                format_func=lambda x: f"{x.title()} plausibility",
                horizontal=True,
                key=f"{mut_str}_level"
            )

            return {"level": level}
        
        case Mutation.NEGATED_EVIDENCE_INJECTION:
            # TODO
            pass

        case Mutation.DATE_NUMBER_JITTER:
            # TODO: enforce at least one selection
            st.session_state[f"{mut_str}_categories"] = st.session_state[f"{mut_str}_categories"]
            categories = st.multiselect(
                "Select the categories to apply jitter to",
                options=["date", "number"],
                format_func=lambda x: f"{x.title()}s",
                key=f"{mut_str}_categories"
            )

            return {"categories": categories}
        
        case Mutation.PASSAGE_SHUFFLE:
            st.session_state[f"{mut_str}_preserve_logical_flow"] = st.session_state[f"{mut_str}_preserve_logical_flow"]
            preserve_logical_flow = st.checkbox(
                "Preserve the logical flow of passages",
                key=f"{mut_str}_preserve_logical_flow"
            )

            return {"preserve_logical_flow": preserve_logical_flow}

        case Mutation.ENTITY_SWAP:
            # TODO: enforce at least one selection
            st.session_state[f"{mut_str}_entity_types"] = st.session_state[f"{mut_str}_entity_types"]
            entity_types = st.multiselect(
                "Select the types of entities to swap",
                options=["names", "locations", "organisations", "dates", "times", "quantities with units"],
                format_func=lambda x: f"{x.title()}",
                key=f"{mut_str}_entity_types"
            )

            st.session_state[f"{mut_str}_number"] = st.session_state[f"{mut_str}_number"]
            number = st.slider(
                "Select the number of entity swaps that should be performed",
                min_value=1,
                max_value=5,
                step=1,
                key=f"{mut_str}_number"
            )

            return {"entity_types": entity_types, "number": number}

        case Mutation.DOCUMENT_SNIPPET_CUT_OFF:
            # TODO
            pass

        case Mutation.UNIT_CONVERSION_REWRITE:
            # TODO: enforce at least one selection
            st.session_state[f"{mut_str}_unit_types"] = st.session_state[f"{mut_str}_unit_types"]
            unit_types = st.multiselect(
                "Select the types of measurements to rewrite",
                options=["distance", "temperate", "time", "mass / weight", "speed", "area", "data storage"],
                format_func=lambda x: f"{x.title()}",
                key=f"{mut_str}_unit_types"
            )

            return {"unit_types": unit_types}
        
        case Mutation.ABLATE_URL_LINKS:
            st.session_state[f"{mut_str}_handling_choice"] = st.session_state[f"{mut_str}_handling_choice"]
            handling_choice = st.radio(
                "Select how to handle URL links",
                options=["remove", "replace"],
                format_func=lambda x: "Remove along with surrounding context" if x == "remove" else "Replace with placeholder",
                key=f"{mut_str}_handling_choice"
            )

            return {"handling_choice": handling_choice}
        
        case _:
            pass

