#from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer, PretrainedConfig, LlamaConfig, GenerationConfig
import torch

class Model(torch.nn.Module):

    def __init__(self, model_name, num_mem_tokens=0):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, low_cpu_mem_usage=True)
        self.generate_mem_tokens(num_mem_tokens)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    def tokenize(self, text, prepend_mem_tokens=False):
        # Add memory tokens to the beginning of the string
        if prepend_mem_tokens and len(self.mem_tokens) > 0:
            text = ''.join(self.mem_tokens) + text
        
        # Tokenize everything and return the ID values in a PyTorch tensor
        return self.tokenizer(text, return_tensors="pt", return_token_type_ids=False)


    def generate_text(self, in_text, num_outputs=1, device='cuda', return_probs=False):
        num_tokens_gen = 4

        inputs = self.tokenize(in_text)
        output_ids = []
        output_tokens = []
        probs = []
        for _ in range(num_outputs):
            in_len = len(inputs['input_ids'][0])

            inputs['input_ids'] = inputs['input_ids'].to(device)

            generation_config = GenerationConfig(min_new_tokens=num_tokens_gen, max_new_tokens=num_tokens_gen, do_sample=True, output_scores=True, return_dict_in_generate=True)

            output = self.model.generate(**inputs, generation_config=generation_config)
            sequences = output['sequences'].squeeze()
            scores = output['scores']
            output_id = [item for item in sequences[in_len:]]
            output_probs = [scores[i].squeeze()[item] for i, item in enumerate(output_id)]

            tokens = self.tokenizer.decode(output_id)
            if len(output_id) == num_tokens_gen:
                output_ids.append(output_id)
                output_tokens.append(tokens)
                probs.append(output_probs)

        output_ids = torch.tensor(output_ids)
        probs = torch.tensor(probs)
        
        if return_probs:
            return (output_ids, probs)
        else:
            return output_tokens


    def generate_mem_tokens(self, num_mem_tokens):
        # Generate tokens of the form [MEM1], [MEM2], etc.
        self.mem_tokens = [f"[MEM{i}]" for i in range(num_mem_tokens)]
        
        # Add the new tokens to the tokenizer
        self.mem_token_ids = []
        for mem_token in self.mem_tokens:
            self.tokenizer.add_tokens(mem_token)
            self.mem_token_ids.append(self.tokenizer.convert_tokens_to_ids(mem_token))

        # Add the tokens to the model
        self.model.resize_token_embeddings(len(self.tokenizer))

        embeddings = self.model.get_input_embeddings()
        for mem_token_id in self.mem_token_ids:
            embeddings.weight[mem_token_id].retain_grad()


    def prep_grad(self):
        if len(self.mem_tokens) > 0:
            embeddings = self.model.get_input_embeddings()
            for i in range(len(self.tokenizer)):
                if i not in self.mem_token_ids:
                    embed_size = embeddings.weight.grad[i,:].shape
                    embeddings.weight.grad[i,:] = torch.zeros(embed_size)

            for name, param in self.model.named_parameters():
                if (name != 'model.embed_tokens.weight') and (param.grad is not None):
                    param.grad.zero_()

    def mem_token_id_mat(self, dim0):
        if len(self.mem_token_ids) > 0:
            mem_token_tuple = ()
            for mem_token_id in self.mem_token_ids:
                mem_token_tuple += (torch.tensor([mem_token_id]*dim0).unsqueeze(1),)

            return torch.cat(mem_token_tuple, dim=1)
        else:
            return torch.tensor([[]]*dim0, dtype=torch.int)
