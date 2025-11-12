# Chain-of-Thought (CoT) Mutations

This directory contains mutation prompts specifically designed for **plain text chain-of-thought reasoning**, as opposed to the tool-based mutations in `prompts/mutations/`.

## Mutation Types

### 1. **entity_swap.jsonl**
**Purpose**: Swap key entities (people, places, organizations) with plausible alternatives.

**Example**:
- Original: "Kevin Costner starred in Field of Dreams..."
- Mutated: "Tom Hanks starred in Field of Dreams..."

**Use Case**: Test if model blindly accepts entity substitutions in reasoning.

### 2. **date_number_jitter.jsonl**
**Purpose**: Introduce small perturbations to dates, numbers, and measurements.

**Example**:
- Original: "Released in 1994 with 280 seats..."
- Mutated: "Released in 1996 with 265 seats..."

**Use Case**: Test sensitivity to numerical accuracy in reasoning.

### 3. **step_shuffle.jsonl**
**Purpose**: Reorder logical steps in the reasoning while keeping content intact.

**Example**:
- Original: "First, I check X. Then, based on X, I conclude Y."
- Mutated: "First, I conclude Y. Based on this, X must be checked."

**Use Case**: Test if model detects logical flow inconsistencies.

### 4. **conclusion_negation.jsonl**
**Purpose**: Keep reasoning intact but change the conclusion to contradict the evidence.

**Example**:
- Original: "...Therefore, the answer is Field of Dreams."
- Mutated: "...Therefore, the answer is Jerry Maguire."

**Use Case**: Test if model blindly follows injected reasoning even when conclusion is wrong.

### 5. **evidence_fabrication.jsonl**
**Purpose**: Add 1-2 plausible but fabricated claims to the reasoning.

**Example**:
- Original: "Timothy Busfield starred in Little Big League..."
- Mutated: "Timothy Busfield starred in Little Big League and Jerry Maguire..."

**Use Case**: Test if model detects hallucinated information in CoT.

### 6. **paraphrase.jsonl** (Control)
**Purpose**: Rephrase the reasoning without changing meaning or facts.

**Example**:
- Original: "The film was released in 1994."
- Mutated: "1994 was the year this movie came out."

**Use Case**: Control mutation to isolate effects of wording changes vs. content changes.

## Usage

These mutations are automatically selected when using the `mutate()` function with `use_cot_prompts=True`. The mutation system will:

1. Load the appropriate prompt template
2. Inject the original CoT and question context
3. Send to LLM for mutation
4. Return the mutated CoT text

## Placeholder Variables

All CoT mutation prompts support:
- `{{cot_text}}`: The original chain-of-thought text to mutate
- `{{question}}`: The original question for context (optional)

## Configuration

To use CoT mutations in experiments, set:
```yaml
mutation_policy: "EntitySwap"  # or any other mutation type
```

The system will automatically use CoT-specific prompts when mutating plain text reasoning.
