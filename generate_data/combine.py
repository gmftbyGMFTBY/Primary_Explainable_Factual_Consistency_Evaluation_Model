import json

data = json.load(open('alpaca_data.json'))
num_alpaca = len(data)
print(f'[!] collect {len(data)} from alpaca_data.json')
data.extend(json.load(open('esnli.json')))
print(f'[!] collect {len(data) - num_alpaca} from esnli.json')
num_1 = len(data)
data.extend(json.load(open('fever.json')))
print(f'[!] collect {len(data) - num_1} from fever.json')

with open('train.json', 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
