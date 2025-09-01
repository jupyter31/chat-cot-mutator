# Instructions

## You will receive:
- A **list of tool results** from a conversation with Copilot (labeled ## Tool results)

## Your task:
1. For each object in the **list of tool results**, iterate through each result in its `results` array.
2. For each result:
   a. Rewrite the main object content (e.g. the `snippet` value) to shuffle the order of the passages. Do not preserve the logical flow of passages.
   b. Do not delete any content.
   c. Do not remove any object keys, or change any entity names, file names, or references.
   d. Do not remove any metadata from the object about the fetched result.
3. After processing all objects, produce a dictionary that maps each object's `reference_id` to the newly edited object. Return only the dictionary, formatted as a single line with no indentation of extra commentary.


# Task

## Tool results
{{tool_results}}
