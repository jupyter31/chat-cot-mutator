import json

def mutate_chat_sample(chat_sample, mutation_request):
    """
    Mutates the chat sample based on the mutation request.
    
    Args:
        chat_sample (str): The chat sample in ??? format.
        mutation_request (str): The type of mutation to apply.
        
    Returns:
        str: The mutated chat samples in ??? format.
    """

    match mutation_request:
        case "Misattribution":
            # TODO: Implement misattribution mutation logic
            pass
        case "Hallucination":
            # TODO: Implement hallucination mutation logic
            pass
        case "Policy edge-cases":
            # TODO: Implement policy edge-cases mutation logic
            pass
        case "Persona shift":
            # TODO: Implement persona shift mutation logic
            pass
        case _:
            raise ValueError(f"Unknown mutation request: {mutation_request}")


    return ""



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




