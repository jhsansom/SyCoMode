from evaluations import test_input_string
import objectives
import exphelper

'''
DESCRIPTION

This experiment is slightly more complex than experiment 1 because there is an additional prompt. The model
must distill the prompt without knowing the future contexts in which that information will be needed.
'''

(args, objective_function) = exphelper.parse_args()
exphelper.wandb_track('basic reasoning', args)

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
    perplexities = test_input_string(args.model_name, 
        ans, 
        in_text, 
        objective_function,
        extra_txt=extra_txt, 
        device=device, 
        lr=args.lr, 
        num_iter=args.num_iter, 
        num_mem_tokens=args.num_mem_tokens)
    orig = perplexities[0]
    no_context = perplexities[1]
    delta_new = perplexities[2]
    print(f'ORIG = {orig:.3f}, WO_CONTEXT = {no_context:.3f}, DISTILLED = {delta_new:.3f}')