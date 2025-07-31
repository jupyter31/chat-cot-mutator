import copy
from deepdiff import DeepDiff
import json

from clients.llm_api import llm_api_client
from mutation_data import get_affected_role, get_mutation_messages


def add_new_responses_to_mutated_chat_samples(mutated_chat_samples, new_responses):
    """
    Adds the new responses to the mutated chat samples.

    Args:
        mutated_chat_samples (list<dict>): The mutated chat samples.
        list<str>: The new responses generated from the mutated chat samples.

    Returns:
        list<dict>: The mutated chat samples with the new responses added.
    """
    for i, chat in enumerate(mutated_chat_samples):
        # remove original assistant response if it exists
        if chat["messages"][-1]["role"] == "assistant":
            chat["messages"].pop()

        # append new response
        chat["messages"].append(
            {
                "role": "assistant",
                "content": new_responses[i],
                "weight": 1
            }
        )

    return mutated_chat_samples


def generate_responses(model, system_prompt, mutated_chat_samples):
    """
    Regenerates the final assistant response using the mutated chat samples.

    Args:
        model (str): The model to use for generating the new responses.
        system_prompt (dict): The system prompt to use for generating new responses.
        mutated_chat_samples (list<dict>): The mutated chat samples.

    Returns:
        list<str>: A list of the new responses of each mutated chat sample.
    """
    # add the system prompt to the mutated messages
    requests = []
    for chat in mutated_chat_samples:
        request = copy.deepcopy(system_prompt)

        # extract just messages, removing original assistant response if it exists
        request["messages"] = request["messages"] + (chat["messages"][:-1] if chat["messages"][-1]["role"] == "assistant" else chat["messages"]) + [{"role":"user", "content":"Answer the user prompt from our message history."}]

        if chat.get("tools") is not None:
            request["tools"] = chat["tools"]

        requests.append(request)

    # send the requests to the LLM API
    responses = llm_api_client.send_batch_chat_request(model, requests)
    return [response["choices"][0]["message"]["content"] for response in responses]


def get_differences(chat_samples, mutated_chat_samples):
    """
    Returns the differences between the original and mutated chat samples.

    Args:
        chat_samples (list<dict>): The original chat samples.
        mutated_chat_samples (list<dict>): The mutated chat samples.

    Returns:
        list<dict>: The differences between the original and mutated chat samples.
    """
    # create copies of the chat samples to avoid modifying the originals
    chat_samples = copy.deepcopy(chat_samples)
    mutated_chat_samples = copy.deepcopy(mutated_chat_samples)

    return [DeepDiff(parse_embedded_json(chat_sample), parse_embedded_json(mutated_chat_sample), view="text") for chat_sample, mutated_chat_sample in zip(chat_samples, mutated_chat_samples)]


def mutate_chat_samples(model, chat_samples, mutation_request, mutation_messages=None):
    """
    Mutates the chat sample based on the mutation request.

    Args:
        model (str): The model to use for inducing the mutations.
        chat_samples (list<dict>): The original chat samples.
        mutation_request (str): The type of mutation to apply.
        mutation_messages (list<dict>, optional): The messages used to perform the mutations.

    Returns:
        list<dict>: The mutated chat samples.
        list<dict>: The messages used to perform the mutations.
    """
    # create a copy of the chat samples to avoid modifying the originals
    chat_samples = copy.deepcopy(chat_samples)

    requests = []

    # get default mutation messages if not provided
    if mutation_messages is None:
        mutation_messages = list(get_mutation_messages(mutation_request))

    # add mutation messages to each chat sample
    for chat in chat_samples:
        requests.append({"messages": chat["messages"] + mutation_messages})

    # send the requests to the LLM API
    responses = llm_api_client.send_batch_chat_request(model, requests)

    affected_role = get_affected_role(mutation_request)

    mutated_chat_samples = []
    if affected_role == "user":
        # replace the original user message with the mutated user message
        for chat, response in zip(chat_samples, responses):
            for msg in chat["messages"]:
                if msg["role"] == affected_role:
                    msg["content"] = response["choices"][0]["message"]["content"]

            mutated_chat_samples.append(chat)

    elif affected_role == "tool":
        # replace content of tool messages with mutated content
        for chat, response in zip(chat_samples, responses):
            try:
                sub_responses = json.loads(response["choices"][0]["message"]["content"])

                for msg in chat["messages"]:
                    if msg["role"] == affected_role and json.loads(msg["content"]).get("results") is not None:
                        msg["content"] = json.loads(msg["content"])

                        for i, result in enumerate(msg["content"]["results"]):
                            # use the reference number from the result to know which values to replace
                            reference_number = str(result["referenceNumber"])
                            if reference_number in sub_responses.keys():
                                msg["content"]["results"][i] = sub_responses[reference_number]

                        msg["content"] = json.dumps(msg["content"])

                mutated_chat_samples.append(chat)

            except Exception:
                raise Exception(f"Sorry, there has been an error in producing the mutations. Please try clicking the 'Submit' button again.")

    return (mutated_chat_samples, mutation_messages)


def parse_embedded_json(obj):
    """
    Parse any embedded JSON strings into JSON objects.

    Args:
        dict / list / str: The object to parse.
    
    Returns:
        dict / list / str: The parsed object with JSON strings converted to JSON objects.
    """
    if isinstance(obj, dict):
        return {k:parse_embedded_json(v) for k,v in obj.items()}
    elif isinstance(obj, list):
        return [parse_embedded_json(x) for x in obj]
    elif isinstance(obj,str):
        try:
            parsed = json.loads(obj)
            return {k:parse_embedded_json(v) for k,v in parsed.items()}
        except:
            return obj
    
    return obj


def run_full_process(model, chat_samples, mutation_request, system_prompt, mutation_messages=None):
    """
    Runs the full process of mutating chat samples, computing differences, and generating new responses.

    Args:
        model (str): The model to use for the inducing the mutations and generating the responses.
        chat_samples (list<dict>): The original chat samples.
        mutation_request (str): The type of mutation to apply.
        system_prompt (dict): The system prompt to use for generating new responses.
        mutation_messages (list<dict>, optional): The messages used to perform the mutations.

    Returns:
        list<dict>: The mutated chat samples.
        list<dict>: The messages used to perform the mutations.
        list<dict>: The differences between the original and mutated chat samples.
        list<str>: The new responses generated from the mutated chat samples.
    """
    mutated_chat_samples, mutation_messages = mutate_chat_samples(model, chat_samples, mutation_request, mutation_messages)
    differences = get_differences(chat_samples, mutated_chat_samples)
    new_responses = generate_responses(model, system_prompt, mutated_chat_samples)
    mutated_chat_samples = add_new_responses_to_mutated_chat_samples(mutated_chat_samples, new_responses)

    return (mutated_chat_samples, mutation_messages, differences, new_responses)


