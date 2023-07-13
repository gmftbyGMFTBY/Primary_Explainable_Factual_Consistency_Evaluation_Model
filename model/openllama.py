from header import *

class OpenLLAMAModel(nn.Module):

    def __init__(self, **args):
        super(OpenLLAMAModel, self).__init__()
        self.args = args

        # init the tokenizer
        self.vocab = LlamaTokenizer.from_pretrained(args['model_path'])
        self.vocab.bos_token_id = 1
        self.vocab.eos_token_id = 2
        self.vocab.pad_token_id = 2

        self.model = LlamaForCausalLM.from_pretrained(args['model_path'], torch_dtype=torch.float16)
        self.model.cuda(torch.cuda.current_device())

    @torch.no_grad()
    def generate_one_sample(self, batch):
        self.model.eval()
        ids = batch['input_ids'].cuda()
        length = len(ids[0])
        model = self.model
        if batch['decoding_method'] == 'greedy':
            output = model.generate(
                input_ids=ids, 
                max_length=length + batch['generate_len'], 
                use_cache=True
            )
        elif batch['decoding_method'] == 'sampling':
            output = model.generate(
                input_ids=ids, 
                do_sample=True,
                top_p=batch['top_p'],
                top_k=batch['top_k'],
                max_length=length + batch['generate_len'], 
                use_cache=True
            )
        else:
            raise Exception(f'[!] Unknow generate method: {batch["decoding_method"]}')

        output = output[0][length:]
        string = self.vocab.decode(output, skip_special_tokens=False)
        string = string.replace('<s>', '').replace('</s>', '')
        return string.strip()

    def forward(self, inputs):
        outputs = self.model(
            input_ids=inputs['input_ids'].cuda(), 
            attention_mask=inputs['attention_mask'].cuda(), 
            labels=inputs['labels'].cuda()
        )
        loss = outputs.loss
        
        # calculate the token accuarcy
        logits = outputs.logits[:, :-1, :].cpu()
        labels = inputs['labels'][:, 1:]
        chosen_tokens = torch.max(logits, dim=-1)[1]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item()+1)
        return loss, gen_acc
 

class OpenLLAMALoRAModel(nn.Module):

    def __init__(self, **args):
        super(OpenLLAMALoRAModel, self).__init__()
        self.args = args

        # init the tokenizer
        self.vocab = AutoTokenizer.from_pretrained(args['model_path'], trust_remote_code=True)
        self.vocab.add_tokens(['<pad>'])
        self.model = AutoModelForCausalLM.from_pretrained(
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
        self.model.resize_token_embeddings(len(self.vocab))

        self.model = prepare_model_for_kbit_training(self.model)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False,
            r=self.args['lora_r'], 
            lora_alpha=self.args['lora_alpha'], 
            lora_dropout=self.args['lora_dropout'],
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def forward(self, inputs):
        outputs = self.model(
            input_ids=inputs['input_ids'].cuda(), 
            attention_mask=inputs['attention_mask'].cuda(), 
            labels=inputs['labels'].cuda()
        )
        loss = outputs.loss
        
        # calculate the token accuarcy
        logits = outputs.logits[:, :-1, :].cpu()
        labels = inputs['labels'][:, 1:]
        chosen_tokens = torch.max(logits, dim=-1)[1]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item()+1)
        return loss, gen_acc
 
