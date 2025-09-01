# Instructions

## You will receive:
- A **list of tool results** from a conversation with Copilot (labeled ## Tool results)

## Your task:
1. For each object in the **list of tool results**, iterate through each result in its `results` array.
2. For each result:
   a. Remove the main content of the tool result (i.e. the main result of the tool call, such as the `snippet` value).
   b. Do not remove any object keys.
   c. Do not remove any metadata from the object about the fetched result.
3. After processing all objects, produce a dictionary that maps each object's `reference_id` to the newly edited object. Return only the dictionary, formatted as a single line with no indentation of extra commentary.


# Task

## Tool results
{{tool_results}}
