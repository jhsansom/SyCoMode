from transformers import AutoTokenizer, AutoModelForCausalLM
from objectives import distill_on_generated_text


#model_name = 'HuggingFaceH4/tiny-random-LlamaForCausalLM'
model_name = 'huggyllama/llama-7b'
device = 'cpu'

model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=False, low_cpu_mem_usage=True)

prompt = "Hello I am a "
inputs = tokenizer(prompt, return_tensors="pt")['input_ids']

distill_on_generated_text(model, tokenizer, prompt, device='cpu')