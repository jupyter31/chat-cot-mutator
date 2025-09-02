<|im_start|>system
# Instructions

## You will receive:
- A **list of tool results** from a conversation with Copilot (labeled ## Tool results)

## Your task:
1. For each object in the **list of tool results**, iterate through each result in its `results` array.
2. For each result:
   a. Add unrelated noise to the content of the result by inserting a couple of sentences between existing sentences.
   b. Do not delete any existing data.
   c. Do not remove any object keys.
3. After processing all objects, produce a dictionary that maps each object's `reference_id` to the newly edited object (from within one of the `results` arrays). Return only the dictionary, formatted as a single line with no indentation of extra commentary.
<|im_end|>

<|im_start|>user
# Task

## Tool results
{{tool_results}}
<|im_end|>
