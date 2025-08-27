import copy
from deepdiff import DeepDiff
import json
import random
import re

from clients.foundry import foundry_client
from clients.llm_api import llm_api_client

from mutation_data import Mutation, get_affected_role, get_mutation_messages


# maximum number of times that the mutation / response generation process will be retried
MAX_RETRY = 1

# file paths for hallucination judge prompts
CLAIMBREAK_PROMPT_FILE = "prompts\\claimbreak.txt"
SCORE_PROMPT_FILE = "prompts\\score_all.txt"


def add_new_responses_to_mutated_chat_samples(mutated_chat_samples, new_responses):
    """
    Adds the new responses to the mutated chat samples.

    Args:
        mutated_chat_samples (list<dict>): The mutated chat samples.
        new_responses (list<str>): The new responses generated from the mutated chat samples.

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
        model (str): The chat model to use for generating the new responses.
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

    # retry response generation for any responses that are None
    retry = MAX_RETRY
    while retry > 0:
        failed_indices = [i for i, r in enumerate(responses) if r is None]
        if not failed_indices:
            break

        retry_requests = [requests[i] for i in failed_indices]
        retry_responses = llm_api_client.send_batch_chat_request(model, retry_requests)

        for i, retry_response in enumerate(retry_responses):
            if retry_response is not None:
                responses[failed_indices[i]] = retry_response
        
        retry -= 1

    return responses


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
    chat_samples_copy = copy.deepcopy(chat_samples)
    mutated_chat_samples_copy = copy.deepcopy(mutated_chat_samples)

    return [DeepDiff(chat_sample, mutated_chat_sample, view="text") for chat_sample, mutated_chat_sample in zip(chat_samples_copy, mutated_chat_samples_copy)]


def get_safe_responses(responses):
    """
    The main cause of errors from LLM responses is the inclusion of an extra trailing curly bracket.
    This aim of this function is the remove extraneous data at the end of JSON objects, if present.

    Args:
        responses (list<str>): The responses to check.

    Returns:
        list<str>: A list of safe responses.
    """
    decoder = json.JSONDecoder()
    safe_responses = []

    for r in responses:
        try:
            decoded = decoder.raw_decode(r)[0] if r else None
            safe_responses.append(json.dumps(decoded))
        except:
            safe_responses.append(None)

    return safe_responses


def call_foundry_client(foundry_token, chat_samples, mutated_chat_samples):
    """
    Returns the URLs of the differences between the original and mutated chat samples.

    Args:
        foundry_token (str): The Foundry token to use for authentication.
        chat_samples (list<dict>): The original chat samples.
        mutated_chat_samples (list<dict>): The mutated chat samples.

    Returns:
        list<str>: The URLs of the differences between the original and mutated chat samples.
    """
    diff_urls = []
    for chat_sample, mutated_chat_sample in zip(chat_samples, mutated_chat_samples):
        diff_url = foundry_client.save_diff(
            foundry_token,
            json.dumps(chat_sample, indent=2),
            json.dumps(mutated_chat_sample, indent=2),
            "Original chat sample", 
            "Mutated chat sample",
        )
        diff_urls.append(diff_url)

    return diff_urls


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
    except:
        return False


def mutate_chat_samples(model, chat_samples, mutation_request, customisations, mutation_messages=None):
    """
    Mutates the chat sample based on the mutation request.

    Args:
        model (str): The chat model to use for inducing the mutations.
        chat_samples (list<dict>): The original chat samples.
        mutation_request (str): The type of mutation to apply.
        customisations (dict): The customisations to apply to the mutation.
        mutation_messages (list<dict>, optional): The messages used to perform the mutations.

    Returns:
        list<dict>: The mutated chat samples.
        list<dict>: The messages used to perform the mutations.
    """
    # create a copy of the chat samples to avoid modifying the originals
    chat_samples_copy = copy.deepcopy(chat_samples)

    # handle the special case of passage shuffle with outer shuffle depth
    if mutation_request == Mutation.PASSAGE_SHUFFLE and customisations.get("shuffle_depth") == "outer":
        for chat in chat_samples_copy:
            for message in chat["messages"]:
                if message["role"] == "tool" and json.loads(message["content"]).get("results") is not None:
                    content = json.loads(message["content"])
                    results = content["results"]
                    random.shuffle(results)
                    content["results"] = results
                    message["content"] = json.dumps(content)

        mutated_chat_samples = chat_samples_copy
        mutation_messages = None

        return (mutated_chat_samples, mutation_messages)

    # handle the more general cases
    requests = []

    # get default mutation messages if not provided
    if mutation_messages is None:
        mutation_messages = list(get_mutation_messages(mutation_request))

    # add mutation messages to each chat sample
    for chat in chat_samples_copy:
        requests.append({"messages": chat["messages"] + mutation_messages})

    # send the requests to the LLM API
    responses = llm_api_client.send_batch_chat_request(model, requests)
    if mutation_request != Mutation.TOPIC_DILUTION:
        safe_responses = get_safe_responses(responses)

    affected_role = get_affected_role(mutation_request)

    mutated_chat_samples = []
    if affected_role == "user":
        # replace the original user message with the mutated user message
        for chat, response in zip(chat_samples_copy, responses):
            for msg in chat["messages"]:
                if msg["role"] == affected_role:
                    msg["content"] = response

            mutated_chat_samples.append(chat)

    elif affected_role == "tool":
        # retry performing mutations for any responses that are not valid JSON
        retry = MAX_RETRY
        while retry > 0:   
            failed_indices = [i for i, r in enumerate(safe_responses) if not is_json_valid(r)]
            if not failed_indices:
                break

            retry_requests = [requests[i] for i in failed_indices]
            retry_responses = llm_api_client.send_batch_chat_request(model, retry_requests)
            safe_retry_resposes = get_safe_responses(retry_responses)

            for i, retry_response in zip(failed_indices, safe_retry_resposes):
                if is_json_valid(retry_response):
                    safe_responses[i] = json.loads(retry_response)

            retry -= 1

        # replace content of tool messages with mutated content
        for i, (chat, response) in enumerate(zip(chat_samples_copy, safe_responses)):
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

    return (mutated_chat_samples, mutation_messages)


def run_full_process(model, chat_samples, mutation_request, customisations, system_prompt, mutation_messages=None):
    """
    Runs the full process of mutating chat samples, computing differences, and generating new responses.

    Args:
        model (str): The chat model to use for the inducing the mutations and generating the responses.
        chat_samples (list<dict>): The original chat samples.
        mutation_request (str): The type of mutation to apply.
        customisations (dict): The customisations to apply to the mutation.
        system_prompt (dict): The system prompt to use for generating new responses.
        mutation_messages (list<dict>, optional): The messages used to perform the mutations.

    Returns:
        list<dict>: The mutated chat samples.
        list<dict>: The messages used to perform the mutations.
        list<dict>: The differences between the original and mutated chat samples.
        list<str>: The new responses generated from the mutated chat samples.
        list<int>: The indices of the chat samples that failed the mutation process.
    """
    print()
    raw_mutated_chat_samples, mutation_messages = mutate_chat_samples(model, chat_samples, mutation_request, customisations, mutation_messages)
    mut_successes = [i for i, chat in enumerate(raw_mutated_chat_samples) if chat is not None]
    errors = {i: f"Mutation failed after {MAX_RETRY} attempts." for i, chat in enumerate(raw_mutated_chat_samples) if chat is None}
    print(len(mut_successes), "mutations succeeded out of", len(raw_mutated_chat_samples))

    raw_differences = get_differences([chat_samples[i] for i in mut_successes], [raw_mutated_chat_samples[i] for i in mut_successes])
    diff_successes = [i for i, diff in zip(mut_successes, raw_differences) if diff]
    errors.update({i: "No differences were found between the original and mutated chat sample." for i, diff in zip(mut_successes, raw_differences) if not diff})
    print(len(diff_successes), "differences computed out of", len(mut_successes))

    raw_responses = generate_responses(model, system_prompt, [raw_mutated_chat_samples[i] for i in diff_successes])
    res_successes = [i for i, response in zip(diff_successes, raw_responses) if response is not None]
    errors.update({i: f"Response generation failed after {MAX_RETRY} attempts." for i, response in zip(diff_successes, raw_responses) if response is None})
    print(len(res_successes), "responses generated out of", len(diff_successes))

    raw_mutated_chat_samples = add_new_responses_to_mutated_chat_samples([raw_mutated_chat_samples[i] for i in res_successes], [raw_responses[diff_successes.index(i)] for i in res_successes])

    mutated_chat_samples = [None] * len(chat_samples)
    differences = [None] * len(chat_samples)
    responses = [None] * len(chat_samples)

    for i in res_successes:
        mutated_chat_samples[i] = raw_mutated_chat_samples[res_successes.index(i)]
        differences[i] = raw_differences[mut_successes.index(i)]
        responses[i] = raw_responses[diff_successes.index(i)]

    return (mutated_chat_samples, mutation_messages, differences, responses, errors)


def run_claimbreak(model, mutated_chat_samples):
    """
    Breaks the new response into a set of claims.

    Args:
        model (str): The reasoning model to use for evaluating the grounding of the new responses.
        mutated_chat_samples (list<dict>): The mutated chat samples.

    Returns:
        claims (list<dict>): The breakdown of claims from the new responses.
    """
    # read claimbreak prompts from file
    with open(CLAIMBREAK_PROMPT_FILE, 'r', encoding='utf-8') as f:
        system_content, user_content = f.read().strip().split("\n")

    claimbreak_requests = []
    for chat in mutated_chat_samples:

        if chat["messages"][0]["role"] == "user":
            user_query = chat["messages"][0]["content"]
        else:
            raise Exception("No user utterance found in mutated chat sample.")
        
        if chat["messages"][-1]["role"] == "assistant":
            assistant_reply = chat["messages"][-1]["content"]
        else:
            raise Exception("No assistant response found in mutated chat sample.")
        
        claimbreak_requests.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": system_content
                    },
                    {
                        "role": "user",
                        "content": user_content.replace("{{Utterance}}", user_query).replace("{{ModelResponse}}", assistant_reply)
                    }
                ]
            }
        )
    claims = llm_api_client.send_batch_chat_request(model, claimbreak_requests)
    return claims


def run_score_all(model, mutated_chat_samples, claims):
    """
    Scores the claims from the new response.

    Args:
        model (str): The reasoning model to use for evaluating the grounding of the new responses.
        mutated_chat_samples (list<dict>): The mutated chat samples.
        claims (list<dict>): The breakdown of claims from the new responses.

    Returns:
        TODO
    """
    # read score_all prompts from file
    with open(SCORE_PROMPT_FILE, 'r', encoding='utf-8') as f:
        system_content, user_content = f.read().strip().split("\n")


    score_all_requests = []
    for chat, claim in zip(mutated_chat_samples, claims):

        if chat["messages"][0]["role"] == "user" and chat["messages"][-1]["role"] == "assistant":
            search_results = chat["messages"][1:-1] if len(chat["messages"]) > 2 else []
        else:
            raise Exception("The user utterance and/or the assistant response was not found in mutated chat sample.")
        
        score_all_requests.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": system_content
                    },
                    {
                        "role": "user",
                        "content": user_content.replace("{{search_results}}", json.dumps(search_results)).replace("{{claims}}", json.dumps(claim))
                    }
                ]
            }
        )

    reasoning_and_scores = llm_api_client.send_batch_chat_request(model, score_all_requests)

    average_scores = []
    for result in reasoning_and_scores:
        result = result.split("\n\n")

        reasoning = result[:-1]
        final_scores = result[-1]
        print(final_scores)

        matches = re.findall(r'Claim \d+:\s*(\d+(?:\.\d+)?)', final_scores)
        scores = [float(score) for score in matches]
        print(scores)

        average_scores.append(round(sum(scores) / len(scores), 2) if len(scores) > 0 else 0.0)

    print()
    print(average_scores)

    return scores

