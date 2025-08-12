import json

input_path = "test_data\\combined_control_conversations_with_system.jsonl"
output_path = "test_data\\trimmed_combined_control_conversations_with_system.jsonl"

with open(input_path, "r", encoding="utf-8") as f:
    chat_samples = f.read().strip().split("\n")[:1000]

with open(output_path, "w", encoding="utf-8") as out_file:
    for i,sample in enumerate(chat_samples):
        try:
            json_sample = json.loads(sample)

            # remove tool guidance system message
            tool_guidance_index = next(
                (i for i, obj in enumerate(json_sample["messages"]) if obj.get("content", "").startswith("# Tool Guidance:\n")),
                -1
            )
            if tool_guidance_index != -1:
                del json_sample["messages"][tool_guidance_index]

            # remove reminder system message
            remember_index = next(
                (i for i, obj in enumerate(json_sample["messages"]) if obj.get("content", "").startswith("**Remember** the following:")),
                -1
            )
            if remember_index != -1:
                del json_sample["messages"][remember_index]

            # remove persona instructions + examples for Enterprise Copilot
            end_of_persona_index = next(
                (i for i, obj in enumerate(json_sample["messages"]) if obj.get("content", "").startswith("\n# Begin!\n")),
                -1
            )
            if end_of_persona_index != -1:
                json_sample["messages"] = json_sample["messages"][end_of_persona_index + 1:]

            out_file.write(json.dumps(json_sample) + "\n")

        except json.JSONDecodeError as e:
            print(i)
            print(f"Error decoding JSON: {e}")
            continue