import json
from llm_api_sample import LLMClient

def mutate_chat_sample(chat_sample, mutation_request):
    """
    Mutates the chat sample based on the mutation request.
    
    Args:
        chat_sample (str): The chat sample in JSON format.
        mutation_request (str): The type of mutation to apply.
        
    Returns:
        json: The mutated chat samples in JSON format.
    """

    messages = json.loads(chat_sample)["messages"]

    match mutation_request:
        case "Salience removal":
            '''
            Salience removal involves deleting the passage whose tokens have the largest attribution with respect to the answer.
            This means that we remove passages from the context that have the largest influence on the answer.
            '''
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

        case "Claim-aligned deletion":
            # TODO
            pass                     
        case "Topic dilution":
            '''
            Topic dilution involves injecting spelling errors, keyboard procimity errors, and visual similarity errors into the chat sample.
            This is done to add noise to the prompt and the tool content.
            '''

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

        case "Negated-evidence injection":
            '''
            Negated-evidence injection involves injecting a passage that contradicts the answer.
            '''
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

        case "Date / number jitter":
            '''
            Date / number jitter involves making date-swap and number-swap edits.
            '''

            request_data = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that introduces date and number jitter to given data."
                    },
                    {
                        "role": "user",
                        "content": f"You are given the messages component of a JSON object used by an LLM. Return the whole JSON object with the user prompt and the tool content rewritten to introduce date and number jitter. Do not return anything else. \n {json.dumps(messages)}"
                    }
                ]
            }
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
        
    response = call_llm_api(request_data)
    print(response)

    mutated_chat_sample = json.loads(chat_sample)
    mutated_chat_sample["messages"] = response

    return [mutated_chat_sample]


def call_llm_api(prompt):
    llm_client = LLMClient(None)

    request_data = prompt
    response = llm_client.send_chat_request("dev-gpt-4o-gg", request_data)

    return response["choices"][0]["message"]["content"]


