# Instructions

## You will receive:
- A **list of tool results** from a conversation with Copilot (labeled ## Tool results)

## Your task:
1. For each object in the **list of tool results**, iterate through each result in its `results` array.
2. For each result:
   a. Apply realistic {{written_categories}} jitter to the main content of the object.
      {{instructions}}
   b. Do not change anything that is not a date or number.\n
   c. Do not remove any object keys.
   d. Do not remove any metadata from the object about the fetched result.
3. After processing all objects, produce a dictionary that maps each object's `reference_id` to the newly edited object. Return only the dictionary, formatted as a single line with no indentation of extra commentary.


# Task

## Tool results
{{tool_results}}
