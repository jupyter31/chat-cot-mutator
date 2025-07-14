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
                        "content": "Return the whole JSON object with the most salient passages of the tool content with respect to the assistant golden answer removed."
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
                        "content": "Return the whole JSON object with the user prompt and the tool content rewritten to include spelling mistakes, keyboard proximity errors, and visual similarity errors."
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
                        "content": "Return the whole JSON object with the salient passages of the tool content with respect to the assistant golden answer replaced with the negation of the assistant golden answer."
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
                        "content": "Return the whole JSON object with the user prompt and the tool content rewritten to introduce date and number jitter."
                    }
                ]
            }
        case "Passage shuffle":
            # TODO
            pass
        case "Entity swap":
            '''
            Entity swaooing involes replacing entities such as names, locations, dates, times, quantities with units, and organisations with a different entity of the same type, while keeping the context and meaning of the conversation intact.
            '''

            request_data = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that performs entity swapping on given data. This involves replacing entities such as names, locations, dates, times, quantities with units, and organisations with a different entity of the same type, while keeping the context and meaning of the conversation intact."
                    },
                    {
                        "role": "user",
                        "content": "Return the whole JSON object with specific entities replaced with different entities of the same type."
                    }
                ]
            }
        case "Document-snippet cut-off":
            # TODO
            pass
        case "Unit-conversion rewrite":
            '''
            Unit-conversion rewrite involves rewriting the chat sample to change the units of measurement to a different unit that measures the same type of quantity, while keeping the numerical values unchanged.
            '''

            request_data = {
                "messages": [
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
                        """
                    },
                    {
                        "role": "user",
                        "content": "Return the whole JSON object whilst changing any units used in the tool content to a different unit that measures the same type of quantity, leaving the numerical value unchanged."
                    }
        ]
    }
        case "Ablate URL links":
            '''
            Ablate URL links involves removing all URLs from the chat sample.
            This means that the LLM does not have the the ability to access these information sources.
            '''

            request_data = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that removes all URLs from given data."
                    },
                    {
                        "role": "user",
                        "content": "Return the whole JSON object with any URLs and their surrounding phrase context removed."
                    }
                ]
            }
        case _:
            raise ValueError(f"Unknown mutation request: {mutation_request}")
    
    # pad the request_data with generic useful information for the LLM
    request_data["messages"][1]["content"] = f"You are given the messages component of a JSON object used by an LLM. {request_data["messages"][1]["content"]} Do not return anything else. \n Messages : {json.dumps(messages)}"
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


