from evaluations import test_input_string
import objectives

'''
DESCRIPTION

This experiment is slightly more complex than experiment 1 because there is an additional prompt. The model
must distill the prompt without knowing the future contexts in which that information will be needed.
'''

# Experimental parameters
num_mem_tokens = 4 # number of memory tokens; 0 to train directly into model weights
objective_function = objectives.distill_on_hidden_layer # objective function for optimization

# Hyperparameters
lr = 0.05 # learning rate
num_iter = 10 # number of gradient descent steps for each training objective

# Model name
model_name = 'huggyllama/llama-7b'
#model_name = 'HuggingFaceH4/tiny-random-LlamaForCausalLM'

with open('JAXA.txt', 'r') as fp:
    in_text = fp.read()

#extra_txt = 'What was the first model of liquid-fuelled launch vehicle indigenously developed in Japan?'
#ans = 'H-II'

extra_txt = 'Which JAXA rocket replaced the M-V?'
ans = 'Epsilon'

print(f'Text = {in_text}')
device = 'cpu'
print(device)
perplexities = test_input_string(model_name, 
    ans, 
    in_text, 
    objective_function,
    extra_txt=extra_txt, 
    device=device, 
    lr=lr, 
    num_iter=num_iter, 
    num_mem_tokens=num_mem_tokens)
orig = perplexities[0]
no_context = perplexities[1]
delta_new = perplexities[2]
print(f'ORIG = {orig:.3f}, WO_CONTEXT = {no_context:.3f}, DISTILLED = {delta_new:.3f}')