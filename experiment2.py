from evaluations import test_input_string
import objectives

'''
DESCRIPTION

This experiment is slightly more complex than experiment 1 because there is an additional prompt. The model
must distill the prompt without knowing the future contexts in which that information will be needed.
'''

# Experimental parameters
num_mem_tokens = 5 # number of memory tokens; 0 to train directly into model weights
objective_function = objectives.distill_on_hidden_layer # objective function for optimization

# Hyperparameters
lr = 0.05 # learning rate
num_iter = 10 # number of gradient descent steps for each training objective

# Model name
model_name = 'huggyllama/llama-7b'
#model_name = 'HuggingFaceH4/tiny-random-LlamaForCausalLM'

in_texts = [
    'The Bohr tower is in Topeka.',
    'My pants are red.',
    'My pants are blue.'
]

ans_s = ['Topeka', 'Red', 'Blue']

extra_txts = [
    'Where is the Bohr tower?',
    'What color are my pants?',
    'What color are my pants?'
]

for i, in_text in enumerate(in_texts):
    extra_txt = extra_txts[i]
    print(f'Text = {in_text}')
    device = 'cpu'
    print(device)
    ans = ans_s[i]
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