from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc

from objectives import causal_language_model, new_training_objective
from evaluations import calculate_perplexity, judge_on_alphabet


def greedy_produce_text(in_text, model, tokenizer, device='cuda'):
    model_inputs = tokenizer(in_text, return_tensors="pt", return_token_type_ids=False)

    in_len = len(model_inputs['input_ids'][0])

    model_inputs['input_ids'] = model_inputs['input_ids'].to(device)

    greedy_output = model.generate(**model_inputs, max_new_tokens=40).squeeze()
    output_ids = [item for item in greedy_output[in_len:]]

    return tokenizer.decode(output_ids)

def generate_text_from_prompt(prompt, model):
    response = model.generate(prompt, max_new_tokens=5, do_sample=False)
    return response

if __name__ == '__main__':
    in_texts = ['My pants are red.',
                'a b c d ',
                'l m n o ',
                't u v w x y ',
                'q r s t ']

    in_texts = [
      'My pants are red.',
      'My pants are blue.'
    ]

    ans_s = ['z', 'e', 'p', 'z', 'u']

    ans_s = ['Red', 'Blue']
    contextless_prompt = 'What color are my pants?'

    for i, in_text in enumerate(in_texts):
        print(f'Text = {in_text}')
        #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        device = 'cpu'
        print(device)
        ans = ans_s[i]
        perplexities = test_input_string(in_text, ans, device, lr=2e-3, num_iter=5, contextless_prompt=contextless_prompt)
        orig = perplexities[0]
        no_context = perplexities[2]
        delta_clm = perplexities[1]
        delta_new = perplexities[3]
        print(f'ORIG = {orig:.3f}, WO_CONTEXT = {no_context:.3f}, CLM = {delta_clm:.3f}, NEW = {delta_new:.3f}')