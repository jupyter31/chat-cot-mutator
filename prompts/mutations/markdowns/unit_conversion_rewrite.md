<|im_start|>system
# Instructions

## You will receive:
- A **list of tool results** from a conversation with Copilot (labeled ## Tool results)

## Your task:
Your task is to swap units with other units that measure the same type of quantity, without changing the numerical value.
Examples:
- Original: The distance is 5 kilometers.
  Modified: The distance is 5 miles.
- Original: The temperature is 20 degrees Celsius.
  Modified: The temperature is 20 degrees Fahrenheit.
- Original: The program took 5ms to run.
  Modified: The program took 5s to run.

1. For each object in the **list of tool results**, iterate through each result in its `results` array.
2. For each result:
   a. Locate any units of measurement that pertain to {{written_unit_types}}.
   b. Replace each unit with a different unit of the same type, keeping the numerical value unchanged. Do not perform any mathematical conversions.
   c. Do not modify anything that is not a unit. Do not modify dates.
   d. Do not remove any object keys.
3. After processing all objects, produce a dictionary that maps each object's `reference_id` to the newly edited object (from within one of the `results` arrays). Return only the dictionary, formatted as a single line with no indentation of extra commentary.
<|im_end|>

<|im_start|>user
# Task

## Tool results
{{tool_results}}
<|im_end|>
