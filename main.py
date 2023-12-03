from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch
import numpy as np
import logging
import tqdm
import math
import gc

# Function for the regular causal language modeling objective
def causal_language_model(model, tokenizer, in_text, lr=1e-5, num_iter=1):
  model.train()

  # Tokenize the input text and convert to tensor
  inputs = tokenizer(in_text, return_tensors="pt")
  input_ids = inputs["input_ids"]

  # Shift the input and label so that the model predicts the next token
  labels = input_ids[..., 1:].contiguous()
  input_ids = input_ids[..., :-1].contiguous()

  # Move tensors to the same device as the model
  input_ids = input_ids.to(model.device)
  labels = labels.to(model.device)

  # Initialize optimizer
  optimizer = Adam(model.parameters(), lr=lr)

  for _ in range(num_iter):

    # Forward pass
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss

    # Backward pass
    loss.backward()

    # Optimization step
    optimizer.step()

    # Clear gradients
    optimizer.zero_grad()

    #print(f"Loss after one step of optimization: {loss.item()}")

def new_training_objective(model, tokenizer, in_text, lr=1e-5, num_iter=1, device='cuda'):
  loss_fn = CrossEntropyLoss()

  # Tokenize the input text and convert to tensor
  inputs = tokenizer(in_text, return_tensors="pt")
  input_ids = inputs["input_ids"]

  # Shift the input and label so that the model predicts the next token
  input_ids_list = [item for item in input_ids.squeeze()]
  labels = input_ids[..., 1:].contiguous()
  input_ids = input_ids[..., :-1].contiguous()

  # Move tensors to the same device as the model
  input_ids = input_ids.to(model.device)

  # Initialize optimizer
  optimizer = Adam(model.parameters(), lr=lr)

  # Forward pass
  with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[:,-1,:]

  inputs = tokenizer("<s>", return_tensors="pt")
  blank_in = inputs["input_ids"].to(device)

  model.train()

  for _ in range(num_iter):
    outputs = model(blank_in)
    new_logits = outputs.logits.squeeze(dim=1)

    #print(logits.shape)
    #print(new_logits.shape)

    loss = loss_fn(new_logits, logits)
    loss.backward()
    optimizer.step()

    # Clear gradients
    optimizer.zero_grad()

def greedy_produce_text(in_text, model, tokenizer, device='cuda'):
  model_inputs = tokenizer(in_text, return_tensors="pt", return_token_type_ids=False)

  in_len = len(model_inputs['input_ids'][0])

  model_inputs['input_ids'] = model_inputs['input_ids'].to(device)

  greedy_output = model.generate(**model_inputs, max_new_tokens=40).squeeze()
  output_ids = [item for item in greedy_output[in_len:]]

  return tokenizer.decode(output_ids)

softmax = torch.nn.Softmax(dim=0)

def calculate_perplexity(model, tokenizer, context, outputs):

  context_ids = tokenizer(context, return_tensors="pt")['input_ids'][0]
  output_ids = tokenizer(outputs, return_tensors="pt")['input_ids'][0]

  full_text = context + ' ' + outputs
  full_ids = tokenizer(full_text, return_tensors="pt", return_token_type_ids=False)

  full_ids['input_ids'] = full_ids['input_ids'].to(device)

  with torch.no_grad():
    outputs = model(**full_ids)
  logits = outputs.logits.squeeze(dim=0)
  logits = softmax(logits)

  grabbed_logits = []
  for i in range(len(output_ids)):
    time_idx = i + len(context_ids) - 1
    token_idx = output_ids[i]
    correct_logit = math.log(logits[time_idx, token_idx].item())
    grabbed_logits.append(correct_logit)

  perplexity_log = - sum(grabbed_logits) / len(grabbed_logits)

  perplexity_lin = math.exp(perplexity_log)

  return perplexity_lin

def test_input_string(in_txt, device, lr=1e-4, num_iter=1):
  # Load tokenizer and model
  #model_name = 'HuggingFaceH4/tiny-random-LlamaForCausalLM'
  model_name = 'huggyllama/llama-7b'
  #model_name = 'abhinavkulkarni/meta-llama-Llama-2-7b-chat-hf-w4-g128-awq'
  tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=False, low_cpu_mem_usage=True)
  #dataset = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
  #quantization = GPTQConfig(bits=4, dataset = dataset, tokenizer=tokenizer)

  loss_fn = CrossEntropyLoss()


  model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)#, quantization_config=quantization)

  model.to(device)

  out_txt = greedy_produce_text(in_txt, model, tokenizer, device=device)
  full_text = in_txt + ' ' + out_txt
  ppl1 = calculate_perplexity(model, tokenizer, in_txt, out_txt)
  #print(f'Initial perplexity = {ppl1}')

  causal_language_model(model, tokenizer, in_txt, lr=lr, num_iter=num_iter)
  ppl2 = calculate_perplexity(model, tokenizer, '<s>', out_txt)
  #print(f'CLM training perplexity = {ppl2}')

  del model
  torch.cuda.empty_cache()
  gc.collect()

  model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)#, quantization_config=quantization)
  model.to(device)

  ppl3 = calculate_perplexity(model, tokenizer, in_txt, out_txt)
  #print(f'Initial perplexity = {ppl1}')

  new_training_objective(model, tokenizer, in_txt, lr=lr, num_iter=num_iter, device=device)
  ppl4 = calculate_perplexity(model, tokenizer, '<s>', out_txt)
  #print(f'New training perplexity = {ppl2}')

  del model
  torch.cuda.empty_cache()
  gc.collect()

  return [ppl1, ppl2, ppl3, ppl4]

if __name__ == '__main__':
    in_texts = ['<s> Hello you are a large language model',
                '<s> This is a test',
                '<s> You are a chat bot',
                '<s> Produce the letter after a b c d ',
                '<s> Produce the letter after l m n o ',
                '<s> Produce the number after 1 2 3 4 ',
                '<s> Produce the number after 4 5 6 7 ']

    for in_text in in_texts:
        print(f'Text = {in_text}')
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print(device)
        perplexities = test_input_string(in_text, device, lr=2e-3, num_iter=10)
        orig = perplexities[0]
        delta_clm = orig - perplexities[1]
        delta_new = orig - perplexities[3]
        print(f'ORIG = {orig:.3f}, CLM = {delta_clm:.3f}, NEW = {delta_new:.3f}')