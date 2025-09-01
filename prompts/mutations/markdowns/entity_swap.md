<|im_start|>system
# Instructions

## You will receive:
- A **list of tool results** from a conversation with Copilot (labeled ## Tool results)

## Your task:
You only swap entities of the following types: {{written_entity_types}}.
1. For each object in the **list of tool results**, iterate through each result in its `results` array.
2. For each result:
   a. Identify the {{number}} most relevant {{entity_plural}} based on its frequency and contextual importance to the original user message and assistant response.
   b. Swap the identified {{entity_plural}} in the context with another of the same type that has appeared in the conversation.
   c. Ensure bidirectional consistency (e.g., if 'Alice' is swapped with 'Bob', also swap 'Bob' with 'Alice').
   d. Ensure consistency of swaps across all messages (e.g., if 'Alice' is swapped with 'Bob' in one message, ensure 'Alice' is swapped with 'Bob' in all messages).
   e. Do not remove any object keys.
3. After processing all objects, produce a dictionary that maps each object's `reference_id` to the newly edited object (from within one of the `results` arrays). Return only the dictionary, formatted as a single line with no indentation of extra commentary.
<|im_end|>

<|im_start|>user
# Task

## Tool results
{{tool_results}}
<|im_end|>
