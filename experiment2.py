from evaluations import test_input_string

'''
DESCRIPTION

This experiment is slightly more complex than experiment 1 because there is an additional prompt. The model
must distill the prompt without knowing the future contexts in which that information will be needed.
'''

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
    perplexities = test_input_string(model_name, ans, in_text, extra_txt=extra_txt, device=device, lr=2e-3, num_iter=5)
    orig = perplexities[0]
    no_context = perplexities[2]
    delta_clm = perplexities[1]
    delta_new = perplexities[3]
    print(f'ORIG = {orig:.3f}, WO_CONTEXT = {no_context:.3f}, CLM = {delta_clm:.3f}, NEW = {delta_new:.3f}')