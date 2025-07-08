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

def validate_json(chat_sample):
    """
    Validates if the provided chat sample is a valid JSON.
    
    Args:
        chat_sample (str): The chat sample in JSON format.
        
    Returns:
        bool: True if valid, False otherwise.
    """
    if chat_sample is not None and chat_sample.strip() != "":
        try:
            chat_sample = json.loads(chat_sample)
            return True
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {e}")
            return False

def remove_escape_characters(chat_sample):
    """
    Removes escape characters from the chat sample.
    
    Args:
        chat_sample (str): The chat sample in JSON format.
        
    Returns:
        str: The chat sample without escape characters.
    """
    return chat_sample.replace("\\", "")




