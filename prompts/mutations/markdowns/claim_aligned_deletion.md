# Instructions

## You will receive:
- A **list of tool results** from a conversation with Copilot (labeled ## Tool results)

## Your task:
1. For each object in the **list of tool results**, iterate through result in its `results` array.
2. For each result:
   a. Split the main content of the object (i.e. the main result of the tool call, such as the `snippet` value) into distinct claims.
   b. Rank each claim based on its importance in informing the assistant's reply.
   c. Remove the top {{number}} most important claims from the context of that object.
   d. Do not remove any object keys.
   e. Do not remove any metadata from the object about the fetched result.
3. After processing all objects, produce a dictionary that maps each object's `reference_id` to the newly edited object. Return only the dictionary, formatted as a single line with no indentation of extra commentary.


# Task

## Tool results
{{tool_results}}
