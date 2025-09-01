# Instructions

## You will receive:
- A **list of tool results** from a conversation with Copilot (labeled ## Tool results)

## Your task:
1. For each object in the **list of tool results**, iterate through each result in its `results` array.
2. For each result:
   a. Split the main content of the object (i.e. the main result of the tool call, such as the `snippet` value) into claims.
   b. Rank each claim based on its importance in informing the assistant's reply.
   c. Rewrite only the 3 most important claims in that object to negate them (i.e. rewrite each claim to contradict it).
   d. Do not delete any claims. Do not delete large chunks of the context.
   e. Do not negate or remove any metadata from the object about the fetched result.
   f. Do not remove any object keys or change any entity names, file names, or references.
3. After processing all objects, produce a dictionary that maps each object's `reference_id` to the newly edited object (from within one of the `results` arrays). Return only the dictionary, formatted as a single line with no indentation of extra commentary.


# Task

## Tool results
{{tool_results}}
