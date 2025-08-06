import json
import streamlit as st

from mutation_data import get_mutation_messages, MUTATION_OPTIONS, DEFAULT_MUTATION_CUSTOMISATIONS


def init_mutation_customisations():
    for mutation_type, params in DEFAULT_MUTATION_CUSTOMISATIONS.items():
        for param, default_value in params.items():
            if f"{mutation_type}_{param}" not in st.session_state:
                st.session_state[f"{mutation_type}_{param}"] = default_value


def edit_mutation_messages():
    with st.expander("Edit messages", expanded=False):

        modified_mutation_messages = st.text_area(
            "Messages",
            value=json.dumps(st.session_state.mutation_messages, indent=2),
            height="content",
            key=f"mutation_messages_{st.session_state.key_suffix}",
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
    mutation_request_selectbox = st.selectbox("Select mutation type", MUTATION_OPTIONS, accept_new_options=False, index=None)
    st.session_state.mutation_request = mutation_request_selectbox if mutation_request_selectbox is not None else st.text_input("Write your own mutation request", placeholder="e.g. 'Rewrite the chat sample with the dates swapped out for different dates.'").strip()
    
    if st.session_state.mutation_request != "":
        # reset customisations and mutation messages to default values
        if st.button("Reset to default", key="reset_mutation_messages"):
            for mutation_type, params in DEFAULT_MUTATION_CUSTOMISATIONS.items():
                for param, default_value in params.items():
                    st.session_state[f"{mutation_type}_{param}"] = default_value

            st.session_state.mutation_messages = list(get_mutation_messages(st.session_state.mutation_request))
            st.session_state.key_suffix += 1

    customisations = get_mutation_customisation()
    st.session_state.mutation_messages = list(get_mutation_messages(st.session_state.mutation_request, customisations))


def get_mutation_customisation():

    st.markdown("##### Mutation customisations")

    match st.session_state.mutation_request:
        # TODO: figure out how this gets its values!?!?
        case "Salience drop":
            number = st.slider(
                "Select the number of salient passages to drop",
                min_value=1,
                max_value=5,
                step=1,
                key=f"{st.session_state.mutation_request}_number_{st.session_state.key_suffix}"
            )

            print(number)
            st.session_state[f"{st.session_state.mutation_request}_number"] = number
            return {"number": number}
        
        case "Claim-aligned deletion":
            # TODO
            pass

        case "Topic dilution":
            level = st.radio(
                "Select the plausibility of the topic dilution",
                options=["high", "medium", "low"],
                index=["high", "medium", "low"].index(DEFAULT_MUTATION_CUSTOMISATIONS["Topic dilution"]["level"]),
                format_func=lambda x: f"{x.title()} plausibility",
                horizontal=True
            )

            return {"level": level}
        
        case "Negated-evidence injection":
            # TODO
            pass

        case "Date / number jitter":
            # TODO: enforce at least one selection
            categories = st.multiselect(
                "Select the categories to apply jitter to",
                options=["date", "number"],
                default=DEFAULT_MUTATION_CUSTOMISATIONS["Date / number jitter"]["categories"],
                format_func=lambda x: f"{x.title()}s",
            )

            return {"categories": categories}
        
        case "Passage shuffle":
            preserve_logical_flow = st.checkbox(
                "Preserve the logical flow of passages",
                value=DEFAULT_MUTATION_CUSTOMISATIONS["Passage shuffle"]["preserve_logical_flow"],
            )

            return {"preserve_logical_flow": preserve_logical_flow}

        case "Entity swap":
            # TODO: enforce at least one selection
            entity_types = st.multiselect(
                "Select the types of entities to swap",
                options=["names", "locations", "organisations", "dates", "times", "quantities with units"],
                default=DEFAULT_MUTATION_CUSTOMISATIONS["Entity swap"]["entity_types"],
                format_func=lambda x: f"{x.title()}",
            )

            number = st.slider(
                "Select the number of entity swaps that should be performed",
                min_value=1,
                max_value=5,
                value=DEFAULT_MUTATION_CUSTOMISATIONS["Entity swap"]["number"],
                step=1,
            )
            
            return {"entity_types": entity_types, "number": number}

        case "Document-snippet cut-off":
            # TODO
            pass

        case "Unit-conversion rewrite":
            # TODO: enforce at least one selection
            unit_types = st.multiselect(
                "Select the types of measurements to rewrite",
                options=["distance", "temperate", "time", "mass / weight", "speed", "area", "data storage"],
                default=DEFAULT_MUTATION_CUSTOMISATIONS["Unit-conversion rewrite"]["unit_types"],
                format_func=lambda x: f"{x.title()}",
            )

            return {"unit_types": unit_types}
        
        case "Ablate URL links":
            # multiple choice of 'remove along with context', 'replace with placeholder'
            handling_choice = st.radio(
                "Select how to handle URL links",
                options=["remove", "replace"],
                index=["remove", "replace"].index(DEFAULT_MUTATION_CUSTOMISATIONS["Ablate URL links"]["handling_choice"]),
                format_func=lambda x: "Remove along with surrounding context" if x == "remove" else "Replace with placeholder",
            )

            return {"handling_choice": handling_choice}
        
        case _:
            pass

