from enum import Enum

class Mutation(Enum):
    SALIENCE_DROP = "Salience drop"
    TOPIC_DILUTION = "Topic dilution"
    NEGATED_EVIDENCE_INJECTION = "Negated-evidence injection"
    DATE_NUMBER_JITTER = "Date / number jitter"
    PASSAGE_SHUFFLE = "Passage shuffle"
    ENTITY_SWAP = "Entity swap"
    UNIT_CONVERSION_REWRITE = "Unit-conversion rewrite"
    ABLATE_URL_LINKS = "Ablate URL links"


MUTATION_MAPPING = {mut: mut.value for mut in Mutation}


DEFAULT_MUTATION_CUSTOMISATIONS = {
    Mutation.SALIENCE_DROP: {"number": 10},
    Mutation.TOPIC_DILUTION: {"level": "high"},
    # TODO: implement negated-evidence injection
    Mutation.DATE_NUMBER_JITTER: {"categories": ["date", "number"]},
    Mutation.PASSAGE_SHUFFLE: {"preserve_logical_flow": False},
    Mutation.ENTITY_SWAP: {"entity_types": ["names"], "number": 1},
    Mutation.UNIT_CONVERSION_REWRITE: {"unit_types": ["time"]},
    Mutation.ABLATE_URL_LINKS: {"handling_choice": "remove"},
}


def get_affected_role(mutation_request):
    """
    Returns the role associated with the specific message we are requesting to mutate.

    Args:
        mutation_request (str): The type of mutation to apply.

    Returns:
        str: The role of the messages that will be modified according to the type of mutation.
    """
    return "user" if mutation_request == Mutation.TOPIC_DILUTION else "tool"


def get_mutation_messages(mutation_request, customisations=None):
    """
    Returns the messages used to perform the mutation.

    Args:
        mutation_request (str): The type of mutation to apply.
        customisations (dict, optional): Customisation parameters for the mutation, if any.

    Returns:
        dict: The system message used to explain the the LLM's role.
        dict: The user message used to perform the mutation.
    """

    if not customisations:
        customisations = DEFAULT_MUTATION_CUSTOMISATIONS.get(mutation_request, {})


    match mutation_request:
        case Mutation.SALIENCE_DROP:
            # Salience drop involves deleting the passage whose tokens have the largest attribution with respect to the answer.
            # This means that we remove passages from the context that have the largest influence on the answer.

            customisations["plural"] = "s" if customisations["number"] > 1 else ""

            return (
                {
                    "role": "system",
                    "content": (
                        "Your task is to process tool-generated messages and extract the most influential content{plural} used in assistant responses."
                    ).format_map(customisations),
                },
                {
                    "role": "user",
                    "content": (
                        "Analyse all tool-generated messages from our conversation containing tool call results.\n"
                        "For each object in the `results` array of each message:\n"
                        "1. Identify the {number} passage{plural} most directly used to inform the assistant's reply.\n"
                        "2. Remove these passage{plural} from the object values, but preserve the object keys.\n"
                        "3. Do not remove any object keys.\n"
                        "Return a dictionary mapping each tool message's `reference_id` (as a string) to its edited object.\n"
                        "Output only the dictionary, formatted as a single line with no indentation or extra commentary."
                    ).format_map(customisations),
                }
            )

        case Mutation.TOPIC_DILUTION:
            # Topic dilution involves injecting spelling errors, keyboard proximity errors, and visual similarity errors into the chat sample.
            # This is done to add noise to the prompt and the tool content.

            return (
                {
                    "role": "system",
                    "content": (
                        "Your task is to introduce spelling, keyboard proximity, and visual similarity errors into user-written text with a {level} plausibility level.\n"
                        "- Keyboard proximity errors occur when adjacent keys are mistakenly pressed.\n"
                        "- Visual similarity errors involve substituting characters that look alike (e.g., '0' for 'O', '1' for 'l')."
                    ).format_map(customisations),
                },
                {
                    "role": "user",
                    "content": (
                        "From our conversation, identify the original user message.\n"
                        "Then:\n"
                        "- Rewrite it with spelling mistakes, keyboard proximity errors, and visual similarity errors with a {level} plausibility level.\n"
                        "- Return only the altered message as a single string, without any commentary or explanation."
                    ).format_map(customisations),
                }
            )

        case Mutation.NEGATED_EVIDENCE_INJECTION:
            # Negated-evidence injection involves injecting a passage that contradicts the answer.

            return (
                {
                    "role": "system",
                    "content": (
                        "Your task is to rewrite tool-generated content by negating key factual claims while preserving correct grammar.\n"
                        "Do not alter entity names, file names, references, or object keys."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "From our conversation, locate all tool-generated messages containing tool call results.\n"
                        "For each object in the `results` array of each message:\n"
                        "1. Identify the factual claims or general statements that were used or paraphrased in the assistant's response.\n"
                        "2. Rewrite the `content` field to negate those claims using appropriate negation (e.g., 'X is true' → 'X is not true').\n"
                        "3. Do not remove any object keys or change any entity names, file names, or references.\n"
                        "Return a dictionary mapping each tool message's `reference_id` (as a string) to its edited object.\n"
                        "Output only the dictionary, formatted as a single line with no indentation or extra commentary."
                    ),
                }
            )

        case Mutation.DATE_NUMBER_JITTER:
            # Date / number jitter involves making date-swap and number-swap edits.

            customisations["written_categories"] = (" and ").join(customisations["categories"])

            customisations["system_message"] = "".join([
                "- Replace dates with plausible alternatives (e.g., past dates with other past dates).\n" if "date" in customisations["categories"] else "",
                "- Replace numbers (e.g., measurements, labels, section numbers) with different but reasonable values." if "number" in customisations["categories"] else ""
            ])

            instructions = []
            if "date" in customisations["categories"]:
                instructions.append("Replace dates with different plausible dates.\n")
            if "number" in customisations["categories"]:
                instructions.append("Replace numbers with different reasonable values.\n")
            instructions.extend("Do not change anything that is not a date or number.\n")
            customisations["user_message"] = "".join(f"{i+1}. {step}" for i, step in enumerate(instructions))

            return (
                {
                    "role": "system",
                    "content": ( 
                        "Your task is to apply realistic {written_categories} jitter to tool-generated content.\n"
                        "{system_message}"
                        "- Do not remove any object keys."
                    ).format_map(customisations),
                },
                {
                    "role": "user",
                    "content": (
                        "From our conversation, locate all tool-generated messages containing tool call results.\n"
                        "For each object in the `results` array of each message:\n"
                        "{user_message}"
                        "Return a dictionary mapping each tool message's `reference_id` (as a string) to its edited object.\n"
                        "Output only the dictionary, formatted as a single line with no indentation or extra commentary."
                    ).format_map(customisations),
                }
            )

        case Mutation.PASSAGE_SHUFFLE:
            # Passage shuffle randomises the passage order to test position bias.

            customisations["preserve"] = "Preserve" if customisations["preserve_logical_flow"] else "Do not preserve"
            customisations["coherent_sense"] = "makes coherent sense" if customisations["preserve_logical_flow"] else "does not make coherent sense"

            return (
                {
                    "role": "system",
                    "content": (
                        "Your task is to randomize and shuffle the order of passages within tool-generated content.\n"
                        "{preserve} the logical flow of the passages so that the shuffled output {coherent_sense}.\n"
                        "Do not remove any object keys or modify entity names, file names, or references."
                    ).format_map(customisations),
                },
                {
                    "role": "user",
                    "content": (
                        "From our conversation, locate all tool-generated messages containing tool call results.\n"
                        "For each object in the `results` array of each message:\n"
                        "1. Identify the factual claims or general statements that were used or paraphrased in the assistant's response.\n"
                        "2. Rewrite the `content` field to shuffle the order of the passages. {preserve} the logical flow of passages.\n"
                        "3. Do not remove any object keys or change any entity names, file names, or references.\n"
                        "Return a dictionary mapping each tool message's `reference_id` (as a string) to its edited object.\n"
                        "Output only the dictionary, formatted as a single line with no indentation or extra commentary."
                    ).format_map(customisations),
                }
            )

        case Mutation.ENTITY_SWAP:
            # Entity swapping involes replacing entities such as names, locations, dates, times, quantities with units, and organisations with a different entity of the same type, while keeping the context and meaning of the conversation intact.

            customisations["written_entity_types"] = (", ").join(customisations["entity_types"])

            customisations["entity_plural"] = "entities" if customisations["number"] > 1 else "entity"

            return (
                {
                    "role": "system",
                    "content": (
                        "Your task is to perform entity swapping on tool-generated content.\n"
                        "- Only swap entities of type: {written_entity_types}. A swap involves replacing an entity with another entity of the same type, and vice versa.\n"
                        "- Use only entities that have already appeared in the conversation.\n"
                        "- Ensure bidirectional consistency (e.g., if 'Alice' is swapped with 'Bob', also swap 'Bob' with 'Alice').\n"
                        "- Ensure consistency of swaps across all messages (e.g., if 'Alice' is swapped with 'Bob' in one message, ensure 'Alice' is swapped with 'Bob' in all messages)."
                    ).format_map(customisations),
                },
                {
                    "role": "user",
                    "content": (
                        "From our conversation, locate all tool-generated messages containing tool call results.\n"
                        "Identify the {number} most relevant {entity_plural} ({written_entity_types}) relevant to the original user message and assistant response.\n"
                        "For each object in the `results` array of each message:\n"
                        "1. Swap the identified {entity_plural} in the tool content with another of the same type that has appeared in the conversation.\n"
                        "2. Ensure entity swaps are consistent across all messages.\n"
                        "3. Do not remove any object keys.\n"
                        "Return a dictionary mapping each tool message's `reference_id` (as a string) to its edited object.\n"
                        "Output only the dictionary, formatted as a single line with no indentation or extra commentary."
                    ).format_map(customisations),
                }
            )

        case Mutation.UNIT_CONVERSION_REWRITE:
            # Unit-conversion rewrite involves rewriting the chat sample to change the units of measurement to a different unit that measures the same type of quantity, while keeping the numerical values unchanged.

            customisations["written_unit_types"] = (", ").join(customisations["unit_types"])

            return (
                {
                    "role": "system",
                    "content": (
                        "Your task is to swap units in tool-generated content with other units that measure the same type of quantity, without changing the numerical value.\n"
                        "- Do not perform any mathematical conversions.\n"
                        "- Only replace the unit (e.g., 'kilometers' → 'miles', 'Celsius' → 'Fahrenheit').\n"
                        "- Ensure the replacement unit is appropriate for the quantity type.\n"
                        "Examples:\n"
                        "- Original: The distance is 5 kilometers.\n"
                        "  Modified: The distance is 5 miles.\n"
                        "- Original: The temperature is 20 degrees Celsius.\n"
                        "  Modified: The temperature is 20 degrees Fahrenheit.\n"
                        "- Original: She is 12 years old.\n"
                        "  Modified: She is 12 months old."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "From our conversation, locate all tool-generated messages containing tool call results.\n"
                        "For each object in the `results` array of each message:\n"
                        "1. Locate any units of measurement that pertain to {written_unit_types}.\n"
                        "2. Replace each unit with a different unit of the same type, keeping the numerical value unchanged.\n"
                        "3. Do not modify anything that is not a unit.\n"
                        "4. Do not remove any object keys."
                        "Return a dictionary mapping each tool message's `reference_id` (as a string) to its edited object.\n"
                        "Output only the dictionary, formatted as a single line with no indentation or extra commentary."
                    ).format_map(customisations),
                }
            )

        case Mutation.ABLATE_URL_LINKS:
            # Ablate URL links involves removing all URLs from the chat sample.
            # This means that the LLM does not have the the ability to access these information sources.

            if customisations["handling_choice"] == "remove":
                customisations["system_message"] = "- Remove the surrounding phrases if necessary to maintain fluency."
                customisations["user_message"] = "Remove all URLs from the `content` field, and adjust surrounding text to maintain grammatical correctness."
            else:
                customisations["system_message"] = "- Replace URLs with a placeholder such as '[URL link removed]'."
                customisations["user_message"] = "1. Replace all URLs in the `content` field with a placeholder such as '[URL link removed]'."

            return (
                {
                    "role": "system",
                    "content": (   
                        "Your task is to remove all URLs from tool-generated content while preserving correct grammar.\n"
                        "- URL links typically start with 'http://', 'https://', or 'www.'.\n"
                        "- URL links typically end with a top-level domain such as '.com', '.org', and '.net'.\n"
                        "{system_message}"
                    ).format_map(customisations),
                },
                {
                    "role": "user",
                    "content": (
                        "From our conversation, locate all tool-generated messages containing tool call results.\n"
                        "For each object in the `results` array of each message:\n"
                        "1. {user_message}\n"
                        "2. Do not remove any object keys.\n"
                        "Return a dictionary mapping each tool message's `reference_id` (as a string) to its edited object.\n"
                        "Output only the dictionary, formatted as a single line with no indentation or extra commentary."
                    ).format_map(customisations),
                }
            )

        case _:
            # Default case for free-form mutation requests

            return (
                {
                    "role": "system",
                    "content": (
                        "Your task is to apply a specified mutation to tool-generated content from our conversation history.\n"
                        "- The mutation will be described in the user message.\n"
                        "- You must not remove any object keys or modify entity names, file names, or references unless explicitly instructed."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "From our conversation, locate all tool-generated messages containing tool call results.\n"
                        "For each object in the `results` array of each message:\n"
                        f"1. Apply the following mutation: {mutation_request}\n"
                        "2. Do not remove any object keys.\n"
                        "Return a dictionary mapping each tool message's `reference_id` (as a string) to its edited object.\n"
                        "Output only the dictionary, formatted as a single line with no indentation or extra commentary."
                    ),
                }
            )
