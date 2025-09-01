# Instructions

## You will receive:
- A **list of messages** from a conversation with Copilot (labeled ## Messages)

## Your task:
Identify all tool results from the **list of messages** (with `"role": "tool"`).
For each object in the `results` array of those message:
1. Apply realistic {{written_categories}} jitter to the main content of the object.
{{instructions}}
2. Do not change anything that is not a date or number.\n
3. Do not remove any object keys.
4. Do not remove any metadata from the object about the fetched result.

Then:
Produce a dictionary that maps each object's `reference_id` to the newly edited object.
Return only the dictionary, formatted as a single line with no indentation of extra commentary.


# Task

## Messages
{{messages}}