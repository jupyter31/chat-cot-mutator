import json

def mutate_chat_sample(chat_sample, mutation_request):
    """
    Mutates the chat sample based on the mutation request.
    
    Args:
        chat_sample (str): The chat sample in JSON format.
        mutation_request (str): The type of mutation to apply.
        
    Returns:
        json: The mutated chat samples in JSON format.
    """
    mutated_chat_sample = json.loads(chat_sample)
    match mutation_request:
        case "Salience removal":
            # TODO
            pass
        case "Claim-aligned deletion":
            for msg in mutated_chat_sample["messages"]:
                if msg["role"] == "tool":
                    msg["content"] = ""
                     
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



def remove_escape_characters(chat_sample):
    """
    Removes escape characters from the chat sample.
    
    Args:
        chat_sample (str): The chat sample in JSON format.
        
    Returns:
        str: The chat sample without escape characters.
    """
    return chat_sample.replace("\\", "")

def add_escape_characters(obj):
    """
    Adds escape characters to the chat sample.
    
    Args:
        chat_sample (dict): The chat sample in JSON format.
        
    Returns:
        dict: The chat sample with escape characters added.
    """

    if isinstance(obj, dict):
        return {k: json.dumps(v) if isinstance(v, dict) else add_escape_characters(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [add_escape_characters(item) for item in obj]
    else:
        return obj




