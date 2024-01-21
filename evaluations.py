from transformers import AutoTokenizer, AutoModelForCausalLM
from model import Model
import torch
import numpy as np
import math
import gc

from objectives import causal_language_model, new_training_objective, distill_on_hidden_layer, distill_on_generated_text

softmax = torch.nn.Softmax(dim=-1)

'''
    Given a full prompt, consisting of context+outputs, this function calculates the perplexity of
    the outputs only (not including the context).
'''
def calculate_perplexity(model, context, outputs, device='cpu', prepend_mem_tokens=False):

    context_ids = model.tokenize(context, prepend_mem_tokens=prepend_mem_tokens)['input_ids'][0]
    output_ids = model.tokenize(outputs, prepend_mem_tokens=prepend_mem_tokens)['input_ids'][0][1:]

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
def judge_on_alphabet(model, context, output, device='cpu', prepend_mem_tokens=False):
    context_ids = model.tokenize(context)['input_ids']

    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 
        'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    correct_idx = np.argwhere(np.array(alphabet) == output).item()

    letters_encoded = model.tokenize(alphabet, prepend_mem_tokens=prepend_mem_tokens)['input_ids']
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
def test_input_string(model_name, 
        ans, 
        in_txt, 
        objective_function, 
        extra_txt='', 
        judgement_func=calculate_perplexity, 
        device='cpu', 
        lr=1e-4, 
        num_iter=1, 
        num_mem_tokens=0
    ):

    # Load model
    model = Model(model_name, num_mem_tokens=num_mem_tokens)
    model.to(device)

    # Get prompts
    if extra_txt != '':
        full_in = in_txt + ' ' + extra_txt
    else:
        full_in = in_txt
    full_text = in_txt + ans

    # Get results before training
    result_fullcontext = judgement_func(model, full_in, ans, device=device)
    result_nocontext = judgement_func(model, extra_txt, ans, device=device)

    # Implement new training objective and measure results
    objective_function(model, in_txt, lr=lr, num_iter=num_iter, device=device, verbose=True)
    result_newmethod = judgement_func(model, extra_txt, ans, device=device, prepend_mem_tokens=True)

    # Get rid of model
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Return all perplexity values
    return [result_fullcontext, result_nocontext, result_newmethod]