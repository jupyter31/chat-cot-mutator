from deepdiff import DeepDiff
from llm_api_client import llm_client

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
        case "Salience removal":
            # Salience removal involves deleting the passage whose tokens have the largest attribution with respect to the answer.
            # This means that we remove passages from the context that have the largest influence on the answer.

            return (
                {
                    "role": "system",
                    "content": "You are a helpful assistant that removes the most salient passage from the tool content that is most relevant to producing the final assistant response.",
                },
                {
                    "role": "user",
                    "content": "Remove the most salient passage with respect to the final assistant response from the tool content in our message history. Return the new tool content in the same JSON format as the original tool content, written on a single line without any indentation.",
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
                    "content": "You are a helpful assistant that rewrites text with spelling errors, keyboard proximity errors, and visual similarity errors. Keyboard proximity errors are errors that occur when a user accidentally types a key that is close to the intended key on the keyboard. Visual similarity errors are errors that occur when a user types a character that looks similar to the intended character, such as '0' instead of 'O' or '1' instead of 'l'.",
                },
                {
                    "role": "user",
                    "content": "Rewrite the user prompt from our message history to include spelling mistakes, keyboard proximity errors, and visual similarity errors. Return the new user prompt.",
                }
            )

        case "Negated-evidence injection":
            # Negated-evidence injection involves injecting a passage that contradicts the answer.

            return (
                {
                    "role": "system",
                    "content": "You are a helpful assistant that negates a given passage whilst maintaining correct grammar.",
                },
                {
                    "role": "user",
                    "content": "Rewrite the tool content from our message history to negate the salient passages that support the final assistant response. Return the new tool content in the same JSON format as the original tool content, written on a single line without any indentation.",
                }
            )

        case "Date / number jitter":
            # Date / number jitter involves making date-swap and number-swap edits.

            return (
                {
                    "role": "system",
                    "content": "You are a helpful assistant that introduces date and number jitter to given data. This involves swapping out dates for different dates and numbers for different numbers.",
                },
                {
                    "role": "user",
                    "content": "Rewrite the tool content from our message history to introduce date and number jitter by replacing dates with different dates and replacing numbers for different numbers. Return the new tool content in the same JSON format as the original tool content, written on a single line without any indentation.",
                }
            )

        case "Passage shuffle":
            # Passage shuffle randomises the passage order to test position bias.

            return (
                {
                    "role": "system",
                    "content": "You are a helpful assistant that shuffles the order of passages in given data.",
                },
                {
                    "role": "user",
                    "content": "Rewrite the tool content from our message history to randomise and shuffle the order of the passages. Return the new tool content in the same JSON format as the original tool content, written on a single line without any indentation.",
                }
            )

        case "Entity swap":
            # Entity swapping involes replacing entities such as names, locations, dates, times, quantities with units, and organisations with a different entity of the same type, while keeping the context and meaning of the conversation intact.

            return (
                {
                    "role": "system",
                    "content": "You are a helpful assistant that performs entity swapping on given data. This involves replacing entities such as names, locations, dates, times, quantities with units, and organisations with a different entity of the same type.",
                },
                {
                    "role": "user",
                    "content": "Swap the entities in the tool content from our message history with other entities of the same type mentioned within our message history. Make sure to include the entities mentioned in the user prompt and assistant response in the swapping. Return the new tool content in the same JSON format as the original tool content, written on a single line without any indentation.",
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
                    "content": """
                        You are a helpful assistant that changes each unit to a different unit that measures the same type of quantity without changing the numerical value. You do not perform mathematical conversions, you simply swap the unit for a different one in the same category.
                        Examples:
                        - Original: The distance is 5 kilometers.
                        - Modified: The distance is 5 miles.

                        - Original: The temperature is 20 degrees Celsius.
                        - Modified: The temperature is 20 degrees Fahrenheit.

                        - Original: She is 12 years old.
                        - Modified: She is 12 months old.
                    """,
                },
                {
                    "role": "user",
                    "content": "Rewrite the tool content from our message history to change any unit mentioned to a different unit that measures the same type of quantity, leaving the numerical value unchanged. Return the new tool content in the same JSON format as the original tool content, written on a single line without any indentation.",
                }
            )

        case "Ablate URL links":
            # Ablate URL links involves removing all URLs from the chat sample.
            # This means that the LLM does not have the the ability to access these information sources.

            return (
                {
                    "role": "system",
                    "content": "You are a helpful assistant that removes all URLs from given data whilst maintaining correct grammar.",
                },
                {
                    "role": "user",
                    "content": "Rewrite the tool content from our message history to remove all URLs and their surrounding phrase context. Return the new tool content in the same JSON format as the original tool content, written on a single line without any indentation.",
                }
            )

        case _:
            # Default case for free-form mutation requests

            return (
                {
                    "role": "system",
                    "content": "You are a helpful assistant that performs the mutation request as specified by the user.",
                },
                {
                    "role": "user",
                    "content": f"Rewrite the tool content from our message history to include the following hallucination: {mutation_request}. Return the new tool content in the same JSON format as the original tool content, written on a single line without any indentation.",
                }
            )


def get_differences(chat_samples, mutated_chat_samples):
    """
    Returns the differences between the original and mutated chat samples.

    Args:
        chat_samples (list<dict>): A list of JSON objects representing individual chat samples.
        mutated_chat_samples (list<dict>): A list of JSON objects representing the mutated chat samples.

    Returns:
        list<dict>: A list of JSON objects representing the differences between the original and mutated chat samples.
    """
    return [DeepDiff(chat_sample, mutated_chat_sample, view="text") for chat_sample, mutated_chat_sample in zip(chat_samples, mutated_chat_samples)]


def mutate_chat_samples(chat_samples, mutation_request):
    """
    Mutates the chat sample based on the mutation request.

    Args:
        chat_samples (list<dict>): A list of JSON objects representing individual chat samples.
        mutation_request (str): The type of mutation to apply.

    Returns:
        list<dict>: A list of JSON objects representing the prompts used to perform the mutations.
        list<dict>: A list of JSON objects representing the mutated chat samples.
    """
    mutation_messages = []
    for sample in chat_samples:
        message_history = {
            "messages": sample["messages"] + list(get_mutation_messages(mutation_request))
        }

        mutation_messages.append(message_history)

    affected_role = get_affected_role(mutation_request)

    responses = llm_client.send_batch_chat_request("dev-gpt-4o-gg", mutation_messages)

    mutated_chat_samples = []
    for chat, response in zip(chat_samples, responses):
        for msg in chat["messages"]:
            if msg["role"] == affected_role:
                msg["content"] = response["choices"][0]["message"]["content"]
                break
        mutated_chat_samples.append(chat)

    return (mutated_chat_samples, mutation_messages)


def mutate_chat_samples_given_prompts(chat_samples, modified_mutation_messages, mutation_request):
    """
    Mutates the chat samples using the provided modified prompts.

    Args:
        chat_samples (list<str>): A list of strings representing individual chat samples.
        modified_mutation_messages (list<dict>): A list of JSON objects representing the modified prompts.
        mutation_request (str): The type of mutation to apply.

    Returns:
        list<dict>: A list of JSON objects representing the mutated chat samples.
    """
    affected_role = get_affected_role(mutation_request)

    responses = llm_client.send_batch_chat_request("dev-gpt-4o-gg", modified_mutation_messages)

    mutated_chat_samples = []
    for chat, response in zip(chat_samples, responses):
        for msg in chat["messages"]:
            if msg["role"] == affected_role:
                msg["content"] = response["choices"][0]["message"]["content"]
                break
        mutated_chat_samples.append(chat)

    return (mutated_chat_samples, modified_mutation_messages)

def regenerate_responses(mutated_chat_samples):
    """
    Regenerates the final assistant response using the mutated chat samples.

    Args:
        mutated_chat_samples (list<list<dict>>): A list of the mutated messages for each chat sample.

    Returns:
        list<str>: A list of the original assistant responses for each chat sample.
        list<str>: A list of the new responses of each mutated chat sample.
    """

    # get list of original assistant responses
    original_responses = [chat["messages"][-1]["content"] if chat["messages"][-1]["role"] == "assistant" else None for chat in mutated_chat_samples]

    # extract just messages, removing original assistant response if it exists
    mutated_messages = [{"messages": chat["messages"][:-1] + [{"role":"user", "content":"Answer the user prompt from our message history."}]} if chat["messages"][-1]["role"] == "assistant" else chat["messages"] for chat in mutated_chat_samples]

    # remove original assistant reponse if it exists
    responses = llm_client.send_batch_chat_request("dev-gpt-4o-gg", mutated_messages)

    new_responses = [response["choices"][0]["message"]["content"] for response in responses]

    return (original_responses, new_responses)
