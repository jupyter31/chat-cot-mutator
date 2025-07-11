import json
from llm_api_sample import LLMClient

def mutate_with_salience_removal(chat_sample):
    """
    Salience removal involves deleting the passage whose tokens have the largest attribution with respect to the answer.
    This means that we remove passages from the context that have the largest influence on the answer.
    """

    mutated_chat_sample = json.loads(chat_sample)
    for msg in mutated_chat_sample["messages"]:
        if msg["role"] == "tool":
            msg["content"] = ""

    return mutated_chat_sample

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
            # TODO
            pass
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





