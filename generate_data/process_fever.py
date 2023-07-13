from datasets import load_dataset
from tqdm import tqdm
import json
import ipdb
import random

result = []
with open('ficle.json') as f:
    dataset = json.load(f)
    for session in dataset:
        claim_span = session['claim_span']
        context_span = session['context_span']
        label = 2
        result.append({
            'instruction': 'Please analyze the factual consistency relationship between claim and context. First, if the relationship is contradiction, generate the claim span and context span. Then, generate the label for factual consistency: emtailment (0), neutral (1), contradiction (2). Note that consistency measures how much information included in the claim is present in the context.',
            f'input': f'=== Context ===\n{session["context"]}\n\n=== Claim ===\n{session["claim"]}',
            f'output': f'1. Claim span: {claim_span}\n2. Context span: {context_span}\n3. Label: {label}'
        })
num_refucts = len(result)


with open('train_fitems.jsonl') as f:
    dataset = [json.loads(line) for line in f.readlines()]

num_entail, num_neutral = 0, 0
for session in tqdm(dataset):
    if session['label'] == 'SUPPORTS':
        label = 0
        claim_span = 'NONE'
        context_span = 'NONE'
        num_entail += 1
    elif session['label'] == 'NOT ENOUGH INFO':
        label = 1
        claim_span = 'NONE'
        context_span = 'NONE'
        num_neutral += 1
    else:
        continue

    if num_entail >= num_refucts and num_neutral >= num_neutral:
        break

    result.append({
        'instruction': 'Please analyze the factual consistency relationship between claim and context. First, if the relationship is contradiction, generate the claim span and context span. Then, generate the label for factual consistency: emtailment (0), neutral (1), contradiction (2). Note that consistency measures how much information included in the claim is present in the context.',
        'input': f'=== Context ===\n{session["context"]}\n\n=== Claim ===\n{session["query"]}',
        'output': f'1. Claim span: {claim_span}\n2. Context span: {context_span}\n3. Label: {label}'
    })

random.shuffle(result)

with open('fever.json', 'w') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
