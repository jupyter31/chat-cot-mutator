def get_affected_role(mutation_request):
    """
    Returns the role associated with the specific message we are requesting to mutate.

    Args:
        mutation_request (str): The type of mutation to apply.

    Returns:
        str: The role associated with the specific message we are requesting to mutate.
    """
    return "user" if mutation_request == "Topic dilution" else "tool"


def get_mutation_messages(mutation_request):
    """
    Returns the messages used to perform the mutation.

    Args:
        mutation_request (str): The type of mutation to apply.

    Returns:
        dict: The system message used to explain the the LLM's role.
        dict: The user message used to perform the mutation.
    """

    match mutation_request:
        case "Salience drop":
            # Salience drop involves deleting the passage whose tokens have the largest attribution with respect to the answer.
            # This means that we remove passages from the context that have the largest influence on the answer.

            return (
                {
                    "role": "system",
                    "content": (
                        "Your task is to process tool-generated messages and extract the most influential content used in assistant responses."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "From our conversation, locate all tool-generated messages containing tool call results.\n"
                        "For each message:\n"
                        "1. Identify the passage most directly used or paraphrased in the assistant's reply.\n"
                        "2. Remove only that salient passage from the `content` field.\n"
                        "3. Do not remove any object keys.\n"
                        "4. Return a dictionary mapping each tool message's `referenceNumber` (as a string) to its edited object.\n"
                        "   Output only the dictionary, formatted as a single line with no indentation or extra commentary."
                    ),
                }
            )

        case "Claim-aligned deletion":
            # TODO
            pass

        case "Topic dilution":
            # Topic dilution involves injecting spelling errors, keyboard procimity errors, and visual similarity errors into the chat sample.
            # This is done to add noise to the prompt and the tool content.

            return (
                {
                    "role": "system",
                    "content": (
                        "Your task is to introduce realistic spelling, keyboard proximity, and visual similarity errors into user-written text.\n"
                        "- Keyboard proximity errors occur when adjacent keys are mistakenly pressed.\n"
                        "- Visual similarity errors involve substituting characters that look alike (e.g., '0' for 'O', '1' for 'l')."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "From our conversation, identify the original user message.\n"
                        "Then:\n"
                        "- Rewrite it with plausible spelling mistakes, keyboard proximity errors, and visual similarity errors.\n"
                        "- Return only the altered message as a single string, without any commentary or explanation."
                    ),
                }
            )

        case "Negated-evidence injection":
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
                        "For each message:\n"
                        "1. Identify the factual claims or general statements that were used or paraphrased in the assistant's response.\n"
                        "2. Rewrite the `content` field to negate those claims using appropriate negation (e.g., 'X is true' → 'X is not true').\n"
                        "3. Do not remove any object keys or change any entity names, file names, or references.\n"
                        "4. Return a dictionary mapping each tool message's `referenceNumber` (as a string) to its edited object.\n"
                        "   Output only the dictionary, formatted as a single line with no indentation or extra commentary."
                    ),
                }
            )

        case "Date / number jitter":
            # Date / number jitter involves making date-swap and number-swap edits.

            return (
                {
                    "role": "system",
                    "content": ( 
                        "Your task is to apply realistic date and number jitter to tool-generated content.\n"
                        "- Replace dates with plausible alternatives (e.g., past dates with other past dates).\n"
                        "- Replace numbers (e.g., measurements, labels, section numbers) with different but reasonable values.\n"
                        "- Do not remove any object keys."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "From our conversation, locate all tool-generated messages containing tool call results.\n"
                        "For each message:\n"
                        "1. Replace dates with different plausible dates.\n"
                        "2. Replace numbers with different reasonable values.\n"
                        "3. Do not change anything that is not a date or number.\n"
                        "4. Return a dictionary mapping each tool message's `referenceNumber` (as a string) to its edited object.\n"
                        "   Output only the dictionary, formatted as a single line with no indentation or extra commentary."
                    ),
                }
            )

        case "Passage shuffle":
            # Passage shuffle randomises the passage order to test position bias.

            return (
                {
                    "role": "system",
                    "content": (
                        "Your task is to randomize and shuffle the order of passages within tool-generated content.\n"
                        "Do not remove any object keys or modify entity names, file names, or references."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "From our conversation, locate all tool-generated messages containing tool call results.\n"
                        "For each message:\n"
                        "1. Identify the factual claims or general statements that were used or paraphrased in the assistant's response.\n"
                        "2. Rewrite the `content` field to shuffle the order of the passages.\n"
                        "3. Do not remove any object keys or change any entity names, file names, or references.\n"
                        "4. Return a dictionary mapping each tool message's `referenceNumber` (as a string) to its edited object.\n"
                        "   Output only the dictionary, formatted as a single line with no indentation or extra commentary."
                    ),
                }
            )

        case "Entity swap":
            # Entity swapping involes replacing entities such as names, locations, dates, times, quantities with units, and organisations with a different entity of the same type, while keeping the context and meaning of the conversation intact.

            return (
                {
                    "role": "system",
                    "content": (
                        "Your task is to perform entity swapping on tool-generated content.\n"
                        "- Replace entities such as names, locations, dates, times, quantities with units, and organizations with other entities of the same type.\n"
                        "- Use only entities that have already appeared in the conversation.\n"
                        "- Ensure consistency of swaps across all messages."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "From our conversation, locate all tool-generated messages containing tool call results.\n"
                        "Then:\n"
                        "1. Identify entities relevant to the original user message and assistant response.\n"
                        "2. Replace each entity in the tool content with another of the same type that has appeared in the conversation.\n"
                        "3. Ensure entity swaps are consistent across all messages.\n"
                        "4. Do not remove any object keys.\n"
                        "5. Return a dictionary mapping each tool message's `referenceNumber` (as a string) to its edited object.\n"
                        "   Output only the dictionary, formatted as a single line with no indentation or extra commentary."
                    ),
                }
            )

        case "Document-snippet cut-off":
            # TODO
            pass

        case "Unit-conversion rewrite":
            # Unit-conversion rewrite involves rewriting the chat sample to change the units of measurement to a different unit that measures the same type of quantity, while keeping the numerical values unchanged.

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
                        "For each message:\n"
                        "1. Locate any units of measurement (e.g., time, distance, temperature, quantity).\n"
                        "2. Replace each unit with a different unit of the same type, keeping the numerical value unchanged.\n"
                        "3. Do not modify anything that is not a unit.\n"
                        "4. Do not remove any object keys."
                        "5. Return a dictionary mapping each tool message's `referenceNumber` (as a string) to its edited object.\n"
                        "   Output only the dictionary, formatted as a single line with no indentation or extra commentary."
                    ),
                }
            )

        case "Ablate URL links":
            # Ablate URL links involves removing all URLs from the chat sample.
            # This means that the LLM does not have the the ability to access these information sources.

            return (
                {
                    "role": "system",
                    "content": (   
                        "Your task is to remove all URLs from tool-generated content while preserving correct grammar.\n"
                        "- You may remove surrounding phrases if necessary to maintain fluency.\n"
                        "- Alternatively, you may replace URLs with a placeholder such as '[link removed]'."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "From our conversation, locate all tool-generated messages containing tool call results.\n"
                        "For each message:\n"
                        "1. Remove all URLs from the `content` field.\n"
                        "2. Adjust surrounding text to maintain grammatical correctness.\n"
                        "3. Do not remove any object keys.\n"
                        "4. Return a dictionary mapping each tool message's `referenceNumber` (as a string) to its edited object.\n"
                        "   Output only the dictionary, formatted as a single line with no indentation or extra commentary."
                    ),
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
                        "For each message:\n"
                        f"1. Apply the following mutation: {mutation_request}\n"
                        "2. Do not remove any object keys.\n"
                        "3. Return a dictionary mapping each tool message's `referenceNumber` (as a string) to its edited object.\n"
                        "   Output only the dictionary, formatted as a single line with no indentation or extra commentary."
                    ),
                }
            )