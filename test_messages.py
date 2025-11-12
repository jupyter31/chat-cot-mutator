from core.pipeline import assemble_messages, PromptTemplates
from core.schema import SampleRecord, FrozenContextRecord, FrozenPassageRecord
import json

# Create a test sample
sample = SampleRecord(
    id='test',
    query='Test question?',
    frozen_context=FrozenContextRecord(passages=[
        FrozenPassageRecord(text='Evidence 1', cite='Source 1')
    ], tool_outputs=[]),
    cot_baseline=None,
    mutation_directive=None,
    grounding_rule=None,
    answer_gold='Test answer',
    meta={}
)

# Load prompts
templates = {
    'A': 'Instructions A\n\nQuestion:\n{query}',
    'C': 'Instructions C\n\nQuestion:\n{query}'
}
prompts = PromptTemplates(templates)

# Test condition C with mutated CoT
mutated_cot = 'Step 1: Analyze evidence\nStep 2: Draw conclusion'
messages = assemble_messages('C', sample, mutated_cot=mutated_cot, prompts=prompts)

print(f'Number of messages: {len(messages)}')
print('\nLast 3 messages:')
for i, msg in enumerate(messages[-3:]):
    print(f'\nMessage {len(messages)-3+i} ({msg["role"]}):')
    content = msg.get('content', '')
    if len(content) > 100:
        print(f'  {content[:100]}...')
    else:
        print(f'  {content}')

# Also test without mutated CoT
messages_no_cot = assemble_messages('C', sample, mutated_cot=None, prompts=prompts)
print(f'\n\nWithout mutated CoT: {len(messages_no_cot)} messages')
