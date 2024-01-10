from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss, KLDivLoss
from torch.optim import SGD
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
  optimizer = SGD(model.parameters(), lr=lr)

  for _ in range(num_iter):

    # Forward pass
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    print(loss)

    # Backward pass
    loss.backward()

    # Optimization step
    optimizer.step()

    # Clear gradients
    optimizer.zero_grad()

    #print(f"Loss after one step of optimization: {loss.item()}")

def new_training_objective(model, tokenizer, in_text, lr=1e-5, num_iter=1, device='cuda'):
  #loss_fn = KLDivLoss(reduction='sum')
  loss_fn = CrossEntropyLoss()

  # Tokenize the input text and convert to tensor
  inputs = tokenizer(in_text, return_tensors="pt")
  input_ids = inputs["input_ids"]

  # Shift the input and label so that the model predicts the next token
  #input_ids_list = [item for item in input_ids.squeeze()]
  #labels = input_ids[..., 1:].contiguous()
  #input_ids = input_ids[..., :-1].contiguous()

  # Move tensors to the same device as the model
  input_ids = input_ids.to(model.device)

  # Initialize optimizer
  optimizer = SGD(model.parameters(), lr=lr)

  # Forward pass
  with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[:,-1,:]
  logits = softmax(logits)

  inputs = tokenizer("", return_tensors="pt")
  blank_in = inputs["input_ids"].to(device)

  model.train()

  for _ in range(num_iter):
    outputs = model(blank_in)
    new_logits = outputs.logits.squeeze(dim=1)
    #new_logits = softmax(new_logits)

    #print(logits.shape)
    #print(new_logits.shape)

    loss = loss_fn(new_logits, logits)
    print(loss)
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

softmax = torch.nn.Softmax(dim=-1)

def calculate_perplexity(model, tokenizer, context, outputs):

  context_ids = tokenizer(context, return_tensors="pt")['input_ids'][0]
  output_ids = tokenizer(outputs, return_tensors="pt")['input_ids'][0][1:]

  full_ids = {'input_ids' : torch.concat((context_ids, output_ids)).unsqueeze(0), 'attention_mask' : torch.ones(len(context_ids) + len(output_ids)).unsqueeze(0)}

  #print('STARTING')
  #print(context)
  #print(outputs)

  full_text = context + outputs
  #full_ids = tokenizer(full_text, return_tensors="pt", return_token_type_ids=False)

  #print(full_ids)
  full_ids['input_ids'] = full_ids['input_ids'].to(device)

  #print(tokenizer.decode(context_ids))
  #print(tokenizer.decode(output_ids))
  #print(tokenizer.decode(full_ids['input_ids'][0]))
  #raise Exception
  #print(len(context_ids) + len(output_ids))
  #print(len(full_ids['input_ids'][0]))
  assert(len(context_ids) + len(output_ids) == len(full_ids['input_ids'][0]))

  with torch.no_grad():
    outputs = model(**full_ids)
  logits = outputs.logits.squeeze(dim=0)
  logits = softmax(logits)

  #print('LEN of logits, context, outputs')
  #print(logits.shape)
  #print(len(context_ids))
  #print(len(output_ids))

  grabbed_logits = []
  for i in range(len(output_ids)):
    time_idx = i + len(context_ids) - 1
    token_idx = output_ids[i]
    correct_logit = math.log(logits[time_idx, token_idx].item())
    grabbed_logits.append(correct_logit)

  #print(grabbed_logits)
  perplexity_log = - sum(grabbed_logits) / len(grabbed_logits)

  perplexity_lin = math.exp(perplexity_log)

  return perplexity_lin

def judge_on_alphabet(model, tokenizer, context, output):
  context_ids = tokenizer(context, return_tensors="pt")['input_ids']

  alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

  correct_idx = np.argwhere(np.array(alphabet) == output).item()

  letters_encoded = tokenizer(alphabet, return_tensors="pt")['input_ids']
  letters_encoded = letters_encoded[:,1]
  
  with torch.no_grad():
    outputs = model(context_ids)
  logits = outputs.logits[:,-1,:].squeeze()
  sub_tensor = torch.ones(len(alphabet))
  for i in range(letters_encoded.shape[0]):
      letter_idx = letters_encoded[i]
      sub_tensor[i] = logits[letter_idx]

  logits = softmax(sub_tensor)
  #print(logits)
  #print(correct_idx)
  return logits[correct_idx]

def generate_text_from_prompt(prompt, model):
    response = model.generate(prompt, max_new_tokens=5, do_sample=False)
    return response

def test_input_string(in_txt, ans, device, lr=1e-4, num_iter=1, prompt2=''):
  # Load tokenizer and model
  #model_name = 'HuggingFaceH4/tiny-random-LlamaForCausalLM'
  model_name = 'huggyllama/llama-7b'
  #model_name = 'huggyllama/llama-13b'
  #model_name = 'lmsys/vicuna-13b-delta-v0'
  #model_name = 'abhinavkulkarni/meta-llama-Llama-2-7b-chat-hf-w4-g128-awq'
  tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=False, low_cpu_mem_usage=True)
  tokenizer.add_bos_token = False
  #dataset = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
  #quantization = GPTQConfig(bits=4, dataset = dataset, tokenizer=tokenizer)

  loss_fn = CrossEntropyLoss()

  ppl1 = 0
  ppl2 = 0
  ppl3 = 0
  ppl4 = 0

  if True:
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)#, quantization_config=quantization)

    if prompt2 is not None:
        full_prompt = torch.tensor([tokenizer(in_txt + ' ' + prompt2)['input_ids']])
        response1 = generate_text_from_prompt(full_prompt, model)
        response1 = tokenizer.decode(response1.flatten())
        print(f'Response with context = {response1}')
    model.to(device)

    #out_txt = greedy_produce_text(in_txt, model, tokenizer, device=device)
    out_txt = ans
    full_text = in_txt + out_txt
    #ppl3 = calculate_perplexity(model, tokenizer, '', out_txt)
    ppl3 = judge_on_alphabet(model, tokenizer, '', out_txt)
    #ppl1 = calculate_perplexity(model, tokenizer, in_txt, out_txt)
    ppl1 = judge_on_alphabet(model, tokenizer, in_txt, out_txt)
    #print(f'Initial perplexity = {ppl1}')

    causal_language_model(model, tokenizer, in_txt, lr=2e-5, num_iter=num_iter)
    #ppl2 = calculate_perplexity(model, tokenizer, '', out_txt)
    ppl2 = judge_on_alphabet(model, tokenizer, '', out_txt)
    #print(f'CLM training perplexity = {ppl2}')

    if prompt2 is not None:
        partial_prompt = torch.tensor([tokenizer(prompt2)['input_ids']])
        response2 = generate_text_from_prompt(partial_prompt, model)
        response2 = tokenizer.decode(response2.flatten())
        print(f'Response after CLM = {response2}')

    del model
    torch.cuda.empty_cache()
    gc.collect()
  
  if True:
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)#, quantization_config=quantization)
    model.to(device)

    #ppl3 = calculate_perplexity(model, tokenizer, in_txt, out_txt)
    #print(f'Initial perplexity = {ppl1}')

    new_training_objective(model, tokenizer, in_txt, lr=8e-2, num_iter=num_iter, device=device)
    #ppl4 = calculate_perplexity(model, tokenizer, '', out_txt)
    ppl4 = judge_on_alphabet(model, tokenizer, '', out_txt)
    #print(f'New training perplexity = {ppl2}')

    if prompt2 is not None:
        response3 = generate_text_from_prompt(partial_prompt, model)
        response3 = tokenizer.decode(response3.flatten())
        print(f'Response after NEW = {response3}')

    del model
    torch.cuda.empty_cache()
    gc.collect()

  return [ppl1, ppl2, ppl3, ppl4]
'''
if __name__ == '__main__':
    in_texts = ['My pants are red.',
                'a b c d ',
                'l m n o ',
                't u v w x y ',
                'q r s t ']

    ans_s = ['z', 'e', 'p', 'z', 'u']

    for i, in_text in enumerate(in_texts):
        print(f'Text = {in_text}')
        #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        device = 'cpu'
        print(device)
        ans = ans_s[i]
        perplexities = test_input_string(in_text, ans, device, lr=2e-3, num_iter=5, prompt2=None)
        orig = perplexities[0]
        no_context = perplexities[2]
        delta_clm = perplexities[1]
        delta_new = perplexities[3]
        print(f'ORIG = {orig:.3f}, WO_CONTEXT = {no_context:.3f}, CLM = {delta_clm:.3f}, NEW = {delta_new:.3f}')

'''

import random
import statistics

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

orig_s = []
no_context_s = []
delta_clm_s = []
delta_new_s = []
num_experiments = 100
for i in range(num_experiments):
    print(f'Epoch {i} =============================================')
    rand_idx = random.randint(0, 21)
    max_len = 25 - rand_idx
    rand_len = random.randint(4, max_len)
    string_to_feed = ''
    for j in range(rand_len):
        string_to_feed += alphabet[rand_idx+j]
        string_to_feed += ' '

    ans = alphabet[rand_idx+rand_len]


    print(f'Input = {string_to_feed}')
    print(f'Correct ans = {ans}')

    device = 'cpu'
    perplexities = test_input_string(string_to_feed, ans, device, lr=2e-3, num_iter=2, prompt2=None)
    orig = perplexities[0].item()
    no_context = perplexities[2].item()
    delta_clm = perplexities[1].item()
    delta_new = perplexities[3].item()

    orig_s.append(orig)
    no_context_s.append(no_context)
    delta_clm_s.append(delta_clm)
    delta_new_s.append(delta_new)

    print(f'W_CONTEXT = {orig:.3f}, NO_CONTEXT = {no_context:.3f}, CLM = {delta_clm:.3f}, NEW = {delta_new:.3f}')
    if i > 0:
        print(f'W_CONTEXT MEAN = {statistics.mean(orig_s)}, STDDEV = {statistics.stdev(orig_s)}')
        print(f'NO_CONTEXT MEAN = {statistics.mean(no_context_s)}, STDDEV = {statistics.stdev(no_context_s)}')
        print(f'CLM MEAN = {statistics.mean(delta_clm_s)}, STDDEV = {statistics.stdev(delta_clm_s)}')
        print(f'NEW MEAN = {statistics.mean(delta_new_s)}, STDDEV = {statistics.stdev(delta_new_s)}')
    print()
