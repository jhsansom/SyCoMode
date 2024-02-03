from evaluations import test_input_string
import objectives
import exphelper

'''
DESCRIPTION

'''

(args, objective_function) = exphelper.parse_args()
exphelper.wandb_track('JAXA reasoning', args)

with open('JAXA.txt', 'r') as fp:
    in_text = fp.read()

#extra_txt = 'What was the first model of liquid-fuelled launch vehicle indigenously developed in Japan?'
#ans = 'H-II'

extra_txt = 'Which JAXA rocket replaced the M-V?'
ans = 'Epsilon'
#ans = 'Beta'

print(f'Text = {in_text}')
device = 'cpu'
print(device)
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