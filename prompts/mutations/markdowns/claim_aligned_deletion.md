# Instructions

## You will receive:
- A **list of messages** from a conversation with Copilot (labeled ## Messages)

## Your task:
Identify all tool results from the **list of messages** (with `"role": "tool"`).
For each object in the `results` array of those message:
1. Split the main content of the object (i.e. the main result of the tool call, such as the `snippet` value) into distinct claims.
2. Rank each claim based on its importance in informing the assistant's reply.
3. Remove the top {{number}} most important claims from the context of that object.
4. Do not remove any object keys.
5. Do not remove any metadata from the object about the fetched result.
6. Repeat with the next object / next tool result message.

Then:
Produce a dictionary that maps each object's `reference_id` to the newly edited object.
Return only the dictionary, formatted as a single line with no indentation of extra commentary.


# Task

## Messages
{{messages}}