from enum import Enum
import json

class Mutation(Enum):
    CLAIM_ALIGNED_DELETION = "Claim-aligned deletion"
    SALIENCE_DROP = "Salience drop"
    TOPIC_DILUTION = "Topic dilution"
    NEGATED_EVIDENCE_INJECTION = "Negated-evidence injection"
    DATE_NUMBER_JITTER = "Date / number jitter"
    PASSAGE_SHUFFLE = "Passage shuffle"
    ENTITY_SWAP = "Entity swap"
    UNIT_CONVERSION_REWRITE = "Unit-conversion rewrite"
    ABLATE_URL_LINKS = "Ablate URL links"


MUTATION_MAPPING = {mut: mut.value for mut in Mutation}


DEFAULT_MUTATION_CUSTOMISATIONS = {
    Mutation.CLAIM_ALIGNED_DELETION: {"number": 5},
    Mutation.TOPIC_DILUTION: {"level": "high"},
    Mutation.DATE_NUMBER_JITTER: {"categories": ["date", "number"]},
    Mutation.PASSAGE_SHUFFLE: {"shuffle_depth": "inner"},
    Mutation.ENTITY_SWAP: {"entity_types": ["names"], "number": 1},
    Mutation.UNIT_CONVERSION_REWRITE: {"unit_types": ["time"]},
    Mutation.ABLATE_URL_LINKS: {"handling_choice": "remove"},
}


def get_affected_role(mutation_request):
    """
    Returns the role associated with the specific message we are requesting to mutate.

    Args:
        mutation_request (str): The type of mutation to apply.

    Returns:
        str: The role of the messages that will be modified according to the type of mutation.
    """
    return "user" if mutation_request == Mutation.TOPIC_DILUTION else "tool"


def get_mutation_messages(mutation_request, customisations=None):
    """
    Returns the messages used to perform the mutation.

    Args:
        mutation_request (str): The type of mutation to apply.
        customisations (dict, optional): Customisation parameters for the mutation, if any.

    Returns:
        dict: The system message used to explain the the LLM's role.
        dict: The user message used to perform the mutation.
    """

    if not customisations:
        customisations = DEFAULT_MUTATION_CUSTOMISATIONS.get(mutation_request, {})


    match mutation_request:
        case Mutation.CLAIM_ALIGNED_DELETION:
            # Claim-aligned deletion involves deleting the the claims from the context that have the greatest importance with regards to the assistant's reply.

            with open('prompts\\mutations\\claim_aligned_deletion.jsonl', 'r', encoding='utf-8') as f:
                system_prompt, user_prompt = [json.loads(prompt) for prompt in f.read().strip().split("\n")]

                return [
                    {
                        "role": "system",
                        "content": system_prompt["content"].replace("{{number}}", str(customisations["number"]))
                    },
                    user_prompt  
                ]
            
        case Mutation.SALIENCE_DROP:
            # Salience drop involves deleting the main content of the tool results, as if the files / emails were empty.

            with open('prompts\\mutations\\salience_drop.jsonl', 'r', encoding='utf-8') as f:
                return [json.loads(prompt) for prompt in f.read().strip().split("\n")]

        case Mutation.TOPIC_DILUTION:
            # Topic dilution involves injecting spelling errors, keyboard proximity errors, and visual similarity errors into the chat sample.
            # This is done to add noise to the prompt and the tool content.

            with open('prompts\\mutations\\topic_dilution.jsonl', 'r', encoding='utf-8') as f:
                system_prompt, user_prompt = [json.loads(prompt) for prompt in f.read().strip().split("\n")]

                return [
                    {
                        "role": "system",
                        "content": system_prompt["content"]
                            .replace("{{level}}", customisations["level"])
                    },
                    user_prompt  
                ]

        case Mutation.NEGATED_EVIDENCE_INJECTION:
            # Negated-evidence injection involves injecting a passage that contradicts the answer.

            with open('prompts\\mutations\\negated_evidence_injection.jsonl', 'r', encoding='utf-8') as f:
                return [json.loads(prompt) for prompt in f.read().strip().split("\n")]

        case Mutation.DATE_NUMBER_JITTER:
            # Date / number jitter involves making date-swap and number-swap edits.

            customisations["written_categories"] = (" and ").join(customisations["categories"])

            customisations["instructions"] = "".join([
                "   - Replace dates with plausible alternatives (e.g., past dates with other past dates).\n" if "date" in customisations["categories"] else "",
                "   - Replace numbers (e.g., measurements, labels, section numbers) with different but reasonable values.\n" if "number" in customisations["categories"] else ""
            ])

            with open('prompts\\mutations\\date_number_jitter.jsonl', 'r', encoding='utf-8') as f:
                system_prompt, user_prompt = [json.loads(prompt) for prompt in f.read().strip().split("\n")]

            return [
                {
                    "role": "system",
                    "content": system_prompt["content"]
                        .replace("{{written_categories}}", customisations["written_categories"])
                        .replace("{{instructions}}", customisations["instructions"])
                },
                user_prompt  
            ]
           

        case Mutation.PASSAGE_SHUFFLE:
            # Passage shuffle randomises the passage order to test position bias.

            with open('prompts\\mutations\\passage_shuffle.jsonl', 'r', encoding='utf-8') as f:
                return [json.loads(prompt) for prompt in f.read().strip().split("\n")]

        case Mutation.ENTITY_SWAP:
            # Entity swapping involes replacing entities such as names, locations, dates, times, quantities with units, and organisations with a different entity of the same type, while keeping the context and meaning of the conversation intact.

            customisations["written_entity_types"] = (", ").join(customisations["entity_types"])

            customisations["entity_plural"] = "entities" if customisations["number"] > 1 else "entity"

            with open('prompts\\mutations\\entity_swap.jsonl', 'r', encoding='utf-8') as f:
                system_prompt, user_prompt = [json.loads(prompt) for prompt in f.read().strip().split("\n")]

            return [
                {
                    "role": "system",
                    "content": system_prompt["content"]
                        .replace("{{written_entity_types}}", customisations["written_entity_types"])
                        .replace("{{entity_plural}}", customisations["entity_plural"])
                        .replace("{{number}}", str(customisations["number"]))
                },
                user_prompt
            ]

        case Mutation.UNIT_CONVERSION_REWRITE:
            # Unit-conversion rewrite involves rewriting the chat sample to change the units of measurement to a different unit that measures the same type of quantity, while keeping the numerical values unchanged.

            customisations["written_unit_types"] = (", ").join(customisations["unit_types"])

            with open('prompts\\mutations\\unit_conversion_rewrite.jsonl', 'r', encoding='utf-8') as f:
                system_prompt, user_prompt = [json.loads(prompt) for prompt in f.read().strip().split("\n")]

            return [
                {
                    "role": "system",
                    "content": system_prompt["content"]
                        .replace("{{written_unit_types}}", customisations["written_unit_types"])
                },
                user_prompt
            ]

        case Mutation.ABLATE_URL_LINKS:
            # Ablate URL links involves removing all URLs from the chat sample.
            # This means that the LLM does not have the the ability to access these information sources.

            if customisations["handling_choice"] == "remove":
                customisations["instruction"] = "Remove all URLs, and adjust surrounding text to maintain grammatical correctness."
            else:
                customisations["instruction"] = "Replace all URLs with the placeholder '[link removed]'."

            with open('prompts\\mutations\\ablate_url_links.jsonl', 'r', encoding='utf-8') as f:
                system_prompt, user_prompt = [json.loads(prompt) for prompt in f.read().strip().split("\n")]

            return [
                {
                    "role": "system",
                    "content": system_prompt["content"]
                        .replace("{{instruction}}", customisations["instruction"])
                },
                user_prompt
            ]

        case _:
            # Default case for free-form mutation requests

            with open('prompts\\mutations\\free_form.jsonl', 'r', encoding='utf-8') as f:
                system_prompt, user_prompt = [json.loads(prompt) for prompt in f.read().strip().split("\n")]

            return [
                {
                    "role": "system",
                    "content": system_prompt["content"]
                        .replace("{{mutation_request}}", mutation_request)
                },
                user_prompt
            ]
