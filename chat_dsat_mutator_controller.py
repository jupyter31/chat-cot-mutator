import copy
from deepdiff import DeepDiff
import json

from clients.llm_api import llm_api_client
from clients.foundry import foundry_client

from mutation_data import get_affected_role, get_mutation_messages


# maximum number of times that the mutation / response generation process will be retried
MAX_RETRY = 5


def add_new_responses_to_mutated_chat_samples(mutated_chat_samples, new_responses):
    """
    Adds the new responses to the mutated chat samples.

    Args:
        mutated_chat_samples (list<dict>): The mutated chat samples.
        list<str>: The new responses generated from the mutated chat samples.

    Returns:
        list<dict>: The mutated chat samples with the new responses added.
    """
    for chat, response in zip(mutated_chat_samples, new_responses):
        # remove original assistant response if it exists
        if chat["messages"][-1]["role"] == "assistant":
            chat["messages"].pop()

        # append new response
        chat["messages"].append(
            {
                "role": "assistant",
                "content": response,
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
    response_contents = [r["choices"][0]["message"]["content"] for r in responses]

    # retry response generation for any responses that are None
    retry = MAX_RETRY
    while retry > 0:
        failed_indices = [i for i, r in enumerate(response_contents) if r is None]
        if not failed_indices:
            break

        retry_requests = [requests[i] for i in failed_indices]
        retry_responses = llm_api_client.send_batch_chat_request(model, retry_requests)
        retry_response_contents = [r["choices"][0]["message"]["content"] for r in retry_responses]

        for i, retry_response in enumerate(retry_response_contents):
            if retry_response is not None:
                response_contents[failed_indices[i]] = retry_response
        
        retry -= 1

    return response_contents


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


def get_diff_urls(chat_samples, mutated_chat_samples):
    """
    Returns the URLs of the differences between the original and mutated chat samples.

    Args:
        chat_samples (list<dict>): The original chat samples.
        mutated_chat_samples (list<dict>): The mutated chat samples.

    Returns:
        list<str>: The URLs of the differences between the original and mutated chat samples.
    """
    return [ 
        foundry_client.save_diff(
            json.dumps(chat_sample, indent=2),
            json.dumps(mutated_chat_sample, indent=2),
            "Original chat sample", 
            "Mutated chat sample",
        ) for chat_sample, mutated_chat_sample in zip(chat_samples, mutated_chat_samples)
    ]


def is_json_valid(s):
    """
    Checks if a string is valid JSON.

    Args:
        s (str): The string to check.

    Returns:
        bool: True if the string is valid JSON, False otherwise.
    """
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


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
    response_contents = [r["choices"][0]["message"]["content"] for r in responses]

    affected_role = get_affected_role(mutation_request)

    mutated_chat_samples = []
    if affected_role == "user":
        # replace the original user message with the mutated user message
        for chat, response in zip(chat_samples, response_contents):
            for msg in chat["messages"]:
                if msg["role"] == affected_role:
                    msg["content"] = response

            mutated_chat_samples.append(chat)

    elif affected_role == "tool":
        # retry performing mutations for any responses that are not valid JSON
        retry = MAX_RETRY
        while retry > 0:   
            failed_indices = [i for i, r in enumerate(response_contents) if not is_json_valid(r)]
            if not failed_indices:
                break

            retry_requests = [requests[i] for i in failed_indices]
            retry_responses = llm_api_client.send_batch_chat_request(model, retry_requests)
            retry_response_contents = [r["choices"][0]["message"]["content"] for r in retry_responses]

            for i, retry_response in zip(failed_indices, retry_response_contents):
                if is_json_valid(retry_response):
                    response_contents[i] = json.loads(retry_response)

            retry -= 1

        # replace content of tool messages with mutated content
        failed_indices = []
        for i, (chat, response) in enumerate(zip(chat_samples, response_contents)):
            try:
                response = json.loads(response)
                for msg in chat["messages"]:
                    if msg["role"] == affected_role and json.loads(msg["content"]).get("results") is not None:
                        msg["content"] = json.loads(msg["content"])

                        for i, result in enumerate(msg["content"]["results"]):
                            # use the reference number from the result to know which values to replace
                            reference_id = str(result["reference_id"])
                            if reference_id in response.keys():
                                msg["content"]["results"][i] = response[reference_id]

                        msg["content"] = json.dumps(msg["content"])

                mutated_chat_samples.append(chat)

            except Exception:
                # put None where the mutation failed
                mutated_chat_samples.append(None)
                failed_indices.append(i)

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
        list<int>: The indices of the chat samples that failed the mutation process.
    """

    raw_mutated_chat_samples, mutation_messages = mutate_chat_samples(model, chat_samples, mutation_request, mutation_messages)
    mut_successes = [i for i, chat in enumerate(raw_mutated_chat_samples) if chat is not None]
    errors = {i: f"Mutation failed after {MAX_RETRY} attempts." for i, chat in enumerate(raw_mutated_chat_samples) if chat is None}

    raw_differences = get_differences([chat_samples[i] for i in mut_successes], [raw_mutated_chat_samples[i] for i in mut_successes])
    diff_successes = [i for i, diff in zip(mut_successes, raw_differences) if diff != {}]
    errors.update({i: "No differences were found between the original and mutated chat sample." for i, diff in zip(mut_successes, raw_differences) if diff == {}})


    raw_responses = generate_responses(model, system_prompt, [raw_mutated_chat_samples[i] for i in diff_successes])
    res_successes = [i for i, response in zip(diff_successes, raw_responses) if response is not None]
    errors.update({i: f"Response generation failed after {MAX_RETRY} attempts." for i, response in zip(diff_successes, raw_responses) if response is None})

    raw_mutated_chat_samples = add_new_responses_to_mutated_chat_samples([raw_mutated_chat_samples[i] for i in res_successes], [raw_responses[diff_successes.index(i)] for i in res_successes])
    raw_diff_urls = get_diff_urls([chat_samples[i] for i in mut_successes], [raw_mutated_chat_samples[diff_successes.index(i)] for i in mut_successes])

    mutated_chat_samples = [None] * len(chat_samples)
    differences = [None] * len(chat_samples)
    diff_urls = [None] * len(chat_samples)
    responses = [None] * len(chat_samples)

    for i in res_successes:
        mutated_chat_samples[i] = raw_mutated_chat_samples[diff_successes.index(i)]

        diff, diff_url = raw_differences[mut_successes.index(i)], raw_diff_urls[mut_successes.index(i)]
        differences[i] = diff if diff else None
        diff_urls[i] = diff_url if diff_url else None

        responses[i] = raw_responses[diff_successes.index(i)]

    return (mutated_chat_samples, mutation_messages, differences, diff_urls, responses, errors)


