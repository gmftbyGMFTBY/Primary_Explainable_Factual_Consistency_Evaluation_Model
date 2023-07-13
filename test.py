from header import *
from config import *
from model import *
from mydatasets import *

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model_path', default='internlm/internlm-7b', type=str)
    parser.add_argument('--delta_model_path', default='ckpt/openllama_evaluation/1', type=str)
    return parser.parse_args()

def main(args):
    args.update({
        'lora_r': 32,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
    })
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args['model_path'],
        load_in_4bit=True,
        max_memory={i: '24576MB' for i in range(torch.cuda.device_count())},
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        ),
        trust_remote_code=True
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        r=args['lora_r'],
        lora_alpha=args['lora_alpha'],
        lora_dropout=args['lora_dropout'],
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
    )

    model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, args['delta_model_path'])
    tokenizer = AutoTokenizer.from_pretrained(args['model_path'], trust_remote_code=True)
    print(f'[!] load model and tokenizer over')
    
    with torch.no_grad():
        prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        instruction = 'Please analyze the factual consistency relationship between claim and context. First, if the relationship is contradiction, generate the claim span and context span. Then, generate your explanations about the factual consistency relationship between claim and context. Finally, generate the label for factual consistency: entailment (0), neutral (1), contradiction (2). Note that consistency measures how much information included in the claim is present in the context.' 
        context = 'Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who served as the 45th president of the United States from 2017 to 2021.'
        claim = 'Donald Trump is 46th president of USA.'
        input = f'=== Context ===\n{context}\n\n=== Claim ===\n{claim}'
        prompt = prompt.format_map({'instruction': instruction, 'input': input})
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        tokens = [tokenizer.bos_token_id] + tokens
        input_ids = torch.LongTensor(tokens).unsqueeze(0).cuda()

        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            return_dict_in_generate=True,
        )
        generation = tokenizer.decode(
            outputs['sequences'][0], 
            skip_special_tokens=True
        )
        print(generation)

if __name__ == "__main__":
    args = vars(parser_args())
    main(args)
