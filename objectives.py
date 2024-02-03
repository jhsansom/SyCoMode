from torch.optim import SGD
from torch.nn import CrossEntropyLoss, KLDivLoss, MSELoss
import torch
from transformers import GenerationConfig
import wandb


def str_to_function(func_str):
    if func_str == 'causal_language_model':
        return causal_language_model
    elif func_str == 'distill_on_output_logits':
        return distill_on_output_logits
    elif func_str == 'distill_on_hidden_layer':
        return distill_on_hidden_layer
    elif func_str == 'distill_on_generated_text':
        return distill_on_generated_text
    else:
        raise Exception(f'Function objectives.{func_str} does not exist')


softmax = torch.nn.Softmax(dim=-1)
log_softmax = torch.nn.Softmax(dim=-1)

# Function for the regular causal language modeling objective
def causal_language_model(model, 
        in_text, 
        lr=1e-5, 
        num_iter=1, 
        verbose=False, 
        device='cuda', 
        prepend_mem_tokens=False,
        **kwargs
    ):

    loss_fn = CrossEntropyLoss()
    
    model.train()

    # Tokenize the input text and convert to tensor
    inputs = model.tokenize(in_text, prepend_mem_tokens=prepend_mem_tokens)
    input_ids = inputs["input_ids"]

    # Shift the input and label so that the model predicts the next token
    labels = input_ids[..., 1:].contiguous()
    input_ids = input_ids[..., :-1].contiguous()
    start_idx = len(model.mem_tokens)

    # Move tensors to the same device as the model
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    # Initialize optimizer
    optimizer = SGD(model.parameters(), lr=lr)

    start_idx = 1

    for i in range(num_iter):

        # Forward pass
        outputs = model(input_ids)

        # Extract the logits computed by the model for the generated words
        new_logits = torch.swapaxes(outputs.logits[:,start_idx:-1], 1, 2)
        correct_ids = input_ids[:,1+start_idx:]

        # Compute the cross entropy loss
        loss = loss_fn(new_logits, correct_ids)
        
        if verbose:
            print(f"Loss after step {i} of optimization: {loss.item()}")

        # Backward pass
        loss.backward()
        model.prep_grad()

        # Optimization step
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()


def distill_on_output_logits(
        model, 
        in_text, 
        lr=1e-5, 
        num_iter=1, 
        device='cuda', 
        verbose=False, 
        prepend_mem_tokens=False,
        **kwargs
    ):

    loss_fn = CrossEntropyLoss()
    #loss_fn = KLDivLoss(reduction="batchmean")
    temp = kwargs["temp"]

    # Tokenize the input text and convert to tensor
    inputs = model.tokenize(in_text)
    input_ids = inputs["input_ids"]

    # Move tensors to the same device as the model
    input_ids = input_ids.to(device)

    # Initialize optimizer
    optimizer = SGD(model.parameters(), lr=lr)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:,-1,:]
        logits = softmax(logits/temp)

    inputs = model.tokenize("", prepend_mem_tokens=prepend_mem_tokens)
    blank_in = inputs["input_ids"].to(device)

    model.train()

    for i in range(num_iter):
        outputs = model(blank_in)
        new_logits = outputs.logits.squeeze(dim=1)/temp
        #new_logits = log_softmax(new_logits)

        loss = loss_fn(new_logits, logits)
        if verbose:
            print(f"Loss after step {i} of optimization: {loss.item()}")
        loss.backward()
        model.prep_grad()

        #for p in model.parameters():
        #    print(p.grad.norm())

        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()

def distill_on_hidden_layer(
        model, 
        in_text, 
        lr=1e-5, 
        num_iter=1, 
        device='cuda', 
        verbose=False, 
        prepend_mem_tokens=False,
        **kwargs
    ):

    loss_fn = MSELoss()

    # Tokenize the input text and convert to tensor
    inputs = model.tokenize(in_text)
    input_ids = inputs["input_ids"]

    # Move tensors to the same device as the model
    input_ids = input_ids.to(device)

    # Initialize optimizer
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=0.01)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        logits = outputs.hidden_states[-1]
        logits = logits[:,-1,:]

    inputs = model.tokenize("", prepend_mem_tokens=prepend_mem_tokens)
    blank_in = inputs["input_ids"].to(device)

    model.train()

    for i in range(num_iter):
        outputs = model(blank_in, output_hidden_states=True)
        new_logits = outputs.hidden_states[-1][:,-1,:]

        loss = loss_fn(new_logits, logits)
        if verbose:
            print(f"Loss after step {i} of optimization: {loss.item()}")
        loss.backward()
        model.prep_grad()
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()


def distill_on_generated_text(
        model, 
        in_text, 
        lr=1e-5, 
        num_iter=1, 
        device='cuda', 
        verbose=False,
        **kwargs
    ):

    #loss_fn = MSELoss()
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=0.01)

    num_outputs = 10

    #model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    with torch.no_grad():
        (gen_ids, probs) = model.generate_text(in_text, num_outputs=num_outputs, device=device, return_probs=True)
        probs = softmax(probs)

    bos_tokens = torch.tensor([model.tokenizer.bos_token_id]*num_outputs).unsqueeze(1)
    mem_token_tensor = model.mem_token_id_mat(num_outputs)
    gen_ids = torch.cat((bos_tokens, mem_token_tensor, gen_ids), dim=1)

    start_idx = mem_token_tensor.shape[1]

    model.train()
    for i in range(num_iter):
        # Feed generated outputs through the model with no context
        outputs = model(gen_ids)

        # Extract the logits computed by the model for the generated words
        new_logits = outputs.logits[:,start_idx:-1]
        gen_ids_flattened = gen_ids[:,1+start_idx:].unsqueeze(-1)
        new_logits = new_logits.gather(-1, gen_ids_flattened).squeeze()

        loss = loss_fn(new_logits, probs)

        if verbose:
            print(f"Loss after step {i} of optimization: {loss.item()}")
        loss.backward()
        model.prep_grad()
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()
