import argparse
import objectives
import wandb

# Configure argument parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-mem-tokens', type=int, default=0)
    parser.add_argument('--objective-function', default='causal_language_model')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--num-iter', type=int, default=5)
    parser.add_argument('--temp', type=float, default=1)
    parser.add_argument('--model-name', default='huggyllama/llama-7b') # 'HuggingFaceH4/tiny-random-LlamaForCausalLM'
    args = parser.parse_args()

    # Process arguments
    objective_function = objectives.str_to_function(args.objective_function)

    return (args, objective_function)

# Configure wandb tracking
def wandb_track(exp_str, args):
    wandb.login()
    run = wandb.init(
        project="sycomode",
        config={
            'experiment': exp_str
        }
    )
    wandb.config.update(args)