import torch
import gc
import random
import statistics

from evaluations import calculate_perplexity, judge_on_alphabet, test_input_string

'''
DESCRIPTION

This is a fairly simple experiment wherein the model is tasked with completing a sequence of letters
in alphabetical order. For instance, the LLM might be provided the sequence "a b c d" and it must subsequently produce
the letter "e" in response.

This experiment uses causal language modeling and our new training objective to distill each context (e.g., "a b c d")
and then tests whether the model then outputs the correct final letter.
'''

model_name = 'huggyllama/llama-7b'
#model_name = 'HuggingFaceH4/tiny-random-LlamaForCausalLM'
learning_rate = 8e-4
num_iter = 5

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

orig_s = []
no_context_s = []
delta_clm_s = []
delta_new_s = []
num_experiments = 100
for i in range(num_experiments):
    print(f'Epoch {i} =============================================')

    # Create a random, alphabetically-ordered string of length between 4 and 25
    rand_idx = random.randint(0, 21)
    max_len = 25 - rand_idx
    rand_len = random.randint(4, max_len)
    string_to_feed = ''
    for j in range(rand_len):
        string_to_feed += alphabet[rand_idx+j]
        string_to_feed += ' '
    ans = alphabet[rand_idx+rand_len]

    # Print out string along with answer
    print(f'Input = {string_to_feed}')
    print(f'Correct ans = {ans}')

    # Meat and potatoes: execute the actual training code on the string identified in this loop iteration
    device = 'cpu'
    probs = test_input_string(model_name, ans, string_to_feed, judgement_func=judge_on_alphabet, device=device, lr=learning_rate, num_iter=num_iter)
    orig = probs[0].item()
    no_context = probs[2].item()
    delta_clm = probs[1].item()
    delta_new = probs[3].item()

    # Append result for this loop iteration to a running list of results
    orig_s.append(orig)
    no_context_s.append(no_context)
    delta_clm_s.append(delta_clm)
    delta_new_s.append(delta_new)

    # Print out statistics thus far
    print(f'W_CONTEXT = {orig:.3f}, NO_CONTEXT = {no_context:.3f}, CLM = {delta_clm:.3f}, NEW = {delta_new:.3f}')
    if i > 0:
        print(f'W_CONTEXT MEAN = {statistics.mean(orig_s)}, STDDEV = {statistics.stdev(orig_s)}')
        print(f'NO_CONTEXT MEAN = {statistics.mean(no_context_s)}, STDDEV = {statistics.stdev(no_context_s)}')
        print(f'CLM MEAN = {statistics.mean(delta_clm_s)}, STDDEV = {statistics.stdev(delta_clm_s)}')
        print(f'NEW MEAN = {statistics.mean(delta_new_s)}, STDDEV = {statistics.stdev(delta_new_s)}')
    print()