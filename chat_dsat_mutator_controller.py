def mutate_chat_sample(chat_sample, mutation_request):
    """
    Mutates the chat sample based on the mutation request.
    
    Args:
        chat_sample (str): The chat sample in ??? format.
        mutation_request (str): The type of mutation to apply.
        
    Returns:
        str: The mutated chat sample in ??? format.
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
