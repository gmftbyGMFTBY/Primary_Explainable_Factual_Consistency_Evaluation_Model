from datasets import load_dataset
import json

data = load_dataset('tathagataraha/ficle')['train']

result = []
for session in data:
    claim = session['Claim']
    context = session['Context']
    if session['Inconsistent Claim Component'].startswith('Target'):
        claim_span = session['Target']
    elif session['Inconsistent Claim Component'].startswith('Relation'):
        claim_span = session['Relation']
    else:
        claim_span = session['Source']
    context_span = session['Inconsistent Context-Span']
    result.append({
        'claim': claim,
        'context': context,
        'claim_span': claim_span,
        'context_span': context_span
    })

with open('ficle.json', 'w') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
    print(f'[!] get {len(result)} samples')
