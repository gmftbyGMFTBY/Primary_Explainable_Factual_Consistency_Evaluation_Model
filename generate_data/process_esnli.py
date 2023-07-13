from datasets import load_dataset
import random
import json

data = load_dataset('esnli')['train']

result = []
for session in data:
    claim = session['hypothesis']
    context = session['premise']
    label = session['label']
    explanations = session['explanation_1']
    result.append({
        'instruction': 'Please explain the factual consistency relationship between claim and context. Note that consistency measures how much information included in the claim is present in the context. Then, generate the label for factual consistency: entailment (0), neutral (1), contradiction (2).',
        'input': f"=== Context ===\n{context}\n\n=== Claim ===\n{claim}",
        'output': f'=== Explanation ===\n{explanations}\n\n=== Label ===\n{label}'
    })

result = random.sample(result, 20000)

with open('esnli.json', 'w') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
