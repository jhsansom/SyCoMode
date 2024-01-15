from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import math
import gc

from objectives import causal_language_model, new_training_objective, distill_on_hidden_layer

softmax = torch.nn.Softmax(dim=-1)

'''
    Given a full prompt, consisting of context+outputs, this function calculates the perplexity of
    the outputs only (not including the context).
'''
def calculate_perplexity(model, tokenizer, context, outputs, device='cpu'):

    context_ids = tokenizer(context, return_tensors="pt")['input_ids'][0]
    output_ids = tokenizer(outputs, return_tensors="pt")['input_ids'][0][1:]

    full_ids = {'input_ids' : torch.concat((context_ids, output_ids)).unsqueeze(0), 'attention_mask' : torch.ones(len(context_ids) + len(output_ids)).unsqueeze(0)}

    full_text = context + outputs
    full_ids['input_ids'] = full_ids['input_ids'].to(device)

    assert(len(context_ids) + len(output_ids) == len(full_ids['input_ids'][0]))

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

'''
    This function computes a probability distribution over possible letter responses and then
    returns the probability of the correct one.
'''
def judge_on_alphabet(model, tokenizer, context, output, device='cpu'):
    context_ids = tokenizer(context, return_tensors="pt")['input_ids']

    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 
        'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

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

    return logits[correct_idx]

'''
    Tests an input string in four scenarios and returns the results. Results are defined by a judgement_func,
    which could measure perplexity, probabilities, or something else.

    Here are the scenarios:
    (1) Full in_txt+extra_txt
    (2) Only extra_txt
    (3) Only extra_txt after performing causal language modeling on in_txt
    (4) Only extra_txt after performing our new objective on in_txt

    Summary of inputs:
    - model_name: model name for download from HuggingFace
    - ans: the correct response to in_txt+extra_text
    - in_txt: this is the text we want to distill into the LLM
    - extra_txt: this is the prompt that we preserve during testing (e.g., it could be a question
        to quiz how well the distillation worked)
    - judgement_func: one of the functions from above (typically calculate_perplexity for long-form answers and
        judge_on_alphabet for single-letter answers)
    - device: for PyTorch ('cpu' or 'cuda')
    - lr: learning rate
    - num_iter: number of gradient descent steps for the causal language modeling and new training objective
'''
def test_input_string(model_name, ans, in_txt, extra_txt='', judgement_func=calculate_perplexity, device='cpu', lr=1e-4, num_iter=1):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=False, low_cpu_mem_usage=True)
    tokenizer.add_bos_token = False

    if extra_txt != '':
        full_in = in_txt + ' ' + extra_txt
    else:
        full_in = in_txt

    # Initialize perplexity values to zero
    result_fullcontext = 0
    result_nocontext = 0
    result_clm = 0
    result_newmethod = 0

    # Load fresh model
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    model.to(device)

    full_text = in_txt + ans
    result_clm = judgement_func(model, tokenizer, extra_txt, ans, device=device)
    result_fullcontext = judgement_func(model, tokenizer, full_in, ans, device=device)

    # Implement causal language modeling and measure results
    causal_language_model(model, tokenizer, in_txt, lr=lr, num_iter=num_iter, verbose=True)
    result_nocontext = judgement_func(model, tokenizer, extra_txt, ans, device=device)

    # Get rid of model so we can download a new one
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Try new training method
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)#, quantization_config=quantization)
    model.to(device)

    # Implement new training objective and measure results
    new_training_objective(model, tokenizer, in_txt, lr=lr, num_iter=num_iter, device=device, verbose=True)
    result_newmethod = judgement_func(model, tokenizer, extra_txt, ans, device=device)

    # Get rid of model again
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Return all perplexity values
    return [result_fullcontext, result_nocontext, result_clm, result_newmethod]