# Instructions

## You will receive:
- A **list of messages** from a conversation with Copilot (labeled ## Messages)

## Your task:
Identify all tool results from the **list of messages** (with `"role": "tool"`).
For each object in the `results` array of those message:
1. Split the main content of the object (i.e. the main result of the tool call, such as the `snippet` value) into claims.
2. Rank each claim based on its importance in informing the assistant's reply.
3. Rewrite only the 5 most important claims in that object to negate them (i.e. rewrite each claim to contradict it).
4. Do not delete any claims.
5. Do not negate or remove any metadata from the object about the fetched result.
6. Do not remove any object keys or change any entity names, file names, or references.
Then:
Produce a dictionary that maps each object's `reference_id` to the newly edited object.
Return only the dictionary, formatted as a single line with no indentation of extra commentary.


# Task

## Messages
{{messages}}
