import json
from llm_api_sample import LLMClient

def mutate_with_salience_removal(chat_sample):
    """
    Salience removal involves deleting the passage whose tokens have the largest attribution with respect to the answer.
    This means that we remove passages from the context that have the largest influence on the answer.
    """

    messages = json.loads(chat_sample)["messages"]
    
    request_data = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that removes the most salient passage from the tool content that is most relevant to producing the assistant golden response."
            },
            {
                "role": "user",
                "content": f"You are given the messages component of a JSON object used by an LLM. Return the whole JSON object with the most salient passages of the tool content with respect to the assistant golden answer removed. Do not return anything else. \n {json.dumps(messages)}"
            }
        ]
    }

    response = call_llm_api(request_data)
    print(response)

    mutated_chat_sample = json.loads(chat_sample)
    mutated_chat_sample["messages"] = response

    return [mutated_chat_sample]

def mutate_with_negated_evidence_injection(chat_sample):
    """
    Negated-evidence injection involves injecting a passage that contradicts the answer.
    """

    messages = json.loads(chat_sample)["messages"]
    
    request_data = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that negates a given passage whilst maintaining correct grammar."
            },
            {
                "role": "user",
                "content": f"You are given the messages component of a JSON object used by an LLM. Return the whole JSON object with the salient passages of the tool content with respect to the assistant golden answer replaced with the negation of the assistant golden answer. Do not return anything else. \n {json.dumps(messages)}"
            }
        ]
    }

    response = call_llm_api(request_data)
    print(response)

    mutated_chat_sample = json.loads(chat_sample)
    mutated_chat_sample["messages"] = response

    return [mutated_chat_sample]

def mutate_with_topic_dilution(chat_sample):
    """
    Topic dilution involves injecting spelling errors, keyboard procimity errors, and visual similarity errors into the chat sample.
    This is done to add noise to the prompt and the tool content.
    """

    messages = json.loads(chat_sample)["messages"]

    request_data = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that rewrites text with spelling errors, keyboard proximity errors, and visual similarity errors. Keyboard proximity errors are errors that occur when a user accidentally types a key that is close to the intended key on the keyboard. Visual similarity errors are errors that occur when a user types a character that looks similar to the intended character, such as '0' instead of 'O' or '1' instead of 'l'."
            },
            {
                "role": "user",
                "content": f"You are given the messages component of a JSON object used by an LLM. Return the whole JSON object with the user prompt and the tool content rewritten to include spelling mistakes, keyboard proximity errors, and visual similarity errors. Do not return anything else. \n {json.dumps(messages)}"
            }
        ]
    }

    response = call_llm_api(request_data)
    print(response)

    mutated_chat_sample = json.loads(chat_sample)
    mutated_chat_sample["messages"] = response

    return [mutated_chat_sample]

def mutate_chat_sample(chat_sample, mutation_request):
    """
    Mutates the chat sample based on the mutation request.
    
    Args:
        chat_sample (str): The chat sample in JSON format.
        mutation_request (str): The type of mutation to apply.
        
    Returns:
        json: The mutated chat samples in JSON format.
    """
    match mutation_request:
        case "Salience removal":
            mutated_chat_sample = mutate_with_salience_removal(chat_sample)
        case "Claim-aligned deletion":
            # TODO
            pass                     
        case "Topic dilution":
            # TODO
            pass
        case "Negated-evidence injection":
            mutated_chat_sample = mutate_with_negated_evidence_injection(chat_sample)
        case "Date / number jitter":
            # TODO 
            pass
        case "Passage shuffle":
            # TODO
            pass
        case "Entity swap":
            # TODO
            pass
        case "Document-snippet cut-off":
            # TODO
            pass
        case "Unit-conversion rewrite":
            # TODO
            pass
        case "Ablate URL links":
            # TODO
            pass
        case _:
            raise ValueError(f"Unknown mutation request: {mutation_request}")
        
    # prompt = f"{[prompt]}"

    return [mutated_chat_sample]


def call_llm_api(prompt):
    llm_client = LLMClient(None)

    request_data = prompt
    response = llm_client.send_chat_request("dev-gpt-4o-gg", request_data)

    return response["choices"][0]["message"]["content"]


