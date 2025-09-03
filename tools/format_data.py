import json
import random

# change input and output file paths to requirements
INPUT_PATH = "test_data\\filename.jsonl"
OUTPUT_PATH = "test_data\\filename_formatted_sample.jsonl"

# set how many chats you want in the randomised sample
SAMPLE_SIZE = 500

with open(INPUT_PATH, "r", encoding="utf-8") as in_file:
    chat_samples = in_file.read().strip().split("\n")
    upper_limit = len(chat_samples)

random_indices = sorted(random.sample(range(0, upper_limit), SAMPLE_SIZE))

print(f"Selected indices: {random_indices}")

random_samples = [chat_samples[i] for i in random_indices]

with open(OUTPUT_PATH, "w", encoding="utf-8") as out_file:
    for idx,sample in enumerate(random_samples):
        try:
            json_sample = json.loads(sample)

            tool_guidance_index = next(
                (i for i, obj in enumerate(json_sample["messages"]) if obj.get("content", "").startswith("# Tool Guidance:\n")),
                -1
            )
            if tool_guidance_index != -1:
                del json_sample["messages"][tool_guidance_index]

            remember_index = next(
                (i for i, obj in enumerate(json_sample["messages"]) if obj.get("content", "").startswith("**Remember** the following:")),
                -1
            )
            if remember_index != -1:
                del json_sample["messages"][remember_index]

            end_of_persona_index = next(
                (i for i, obj in enumerate(json_sample["messages"]) if obj.get("content", "").startswith("\n# Begin!\n")),
                -1
            )
            if end_of_persona_index != -1:
                json_sample["messages"] = json_sample["messages"][end_of_persona_index + 1:]

            out_file.write(json.dumps(json_sample) + "\n")

        except json.JSONDecodeError as e:
            print(idx)
            print(f"Error decoding JSON: {e}")
            continue