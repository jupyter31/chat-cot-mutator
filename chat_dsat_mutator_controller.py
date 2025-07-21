from deepdiff import DeepDiff
from llm_api_client import llm_client
from mutation_data import get_affected_role, get_mutation_messages


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
