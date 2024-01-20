from torch.optim import SGD
from torch.nn import CrossEntropyLoss, KLDivLoss, MSELoss
import torch
from transformers import GenerationConfig

softmax = torch.nn.Softmax(dim=-1)

# Function for the regular causal language modeling objective
def causal_language_model(model, tokenizer, in_text, lr=1e-5, num_iter=1, verbose=False):
    model.train()

    # Tokenize the input text and convert to tensor
    inputs = tokenizer(in_text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Shift the input and label so that the model predicts the next token
    labels = input_ids[..., 1:].contiguous()
    input_ids = input_ids[..., :-1].contiguous()

    # Move tensors to the same device as the model
    input_ids = input_ids.to(model.device)
    labels = labels.to(model.device)

    # Initialize optimizer
    optimizer = SGD(model.parameters(), lr=lr)

    for i in range(num_iter):

        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        
        if verbose:
            print(f"Loss after step {i} of optimization: {loss.item()}")

        # Backward pass
        loss.backward()

        # Optimization step
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()


def new_training_objective(model, tokenizer, in_text, lr=1e-5, num_iter=1, device='cuda', verbose=False):
    loss_fn = CrossEntropyLoss()

    # Tokenize the input text and convert to tensor
    inputs = tokenizer(in_text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Move tensors to the same device as the model
    input_ids = input_ids.to(model.device)

    # Initialize optimizer
    optimizer = SGD(model.parameters(), lr=lr)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:,-1,:]
        logits = softmax(logits)

    inputs = tokenizer("", return_tensors="pt")
    blank_in = inputs["input_ids"].to(device)

    model.train()

    for i in range(num_iter):
        outputs = model(blank_in)
        new_logits = outputs.logits.squeeze(dim=1)
        #new_logits = softmax(new_logits)

        loss = loss_fn(new_logits, logits)
        if verbose:
            print(f"Loss after step {i} of optimization: {loss.item()}")
        loss.backward()
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()

def distill_on_hidden_layer(model, tokenizer, in_text, lr=1e-5, num_iter=1, device='cuda', verbose=False):
    loss_fn = MSELoss()

    # Tokenize the input text and convert to tensor
    inputs = tokenizer(in_text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Move tensors to the same device as the model
    input_ids = input_ids.to(model.device)

    # Initialize optimizer
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=0.01)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        logits = outputs.hidden_states[-1]
        logits = logits[:,-1,:]

    inputs = tokenizer("", return_tensors="pt")
    blank_in = inputs["input_ids"].to(device)

    model.train()

    for i in range(num_iter):
        outputs = model(blank_in, output_hidden_states=True)
        new_logits = outputs.hidden_states[-1][:,-1,:]

        loss = loss_fn(new_logits, logits)
        if verbose:
            print(f"Loss after step {i} of optimization: {loss.item()}")
        loss.backward()
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()

def produce_text(model, tokenizer, in_text, num_outputs=1, device='cuda', return_probs=False):
    num_gen = 20
    model_inputs = tokenizer(in_text, return_tensors="pt", return_token_type_ids=False)
    output_ids = []
    output_tokens = []
    probs = []
    for _ in range(num_outputs):
        in_len = len(model_inputs['input_ids'][0])

        model_inputs['input_ids'] = model_inputs['input_ids'].to(device)

        generation_config = GenerationConfig(min_new_tokens=num_gen, max_new_tokens=num_gen, do_sample=True, output_scores=True, return_dict_in_generate=True)

        greedy_output = model.generate(**model_inputs, generation_config=generation_config)
        sequences = greedy_output['sequences'].squeeze()
        scores = greedy_output['scores']
        output_id = [item for item in sequences[in_len:]]
        output_probs = [scores[i].squeeze()[item] for i, item in enumerate(output_id)]

        tokens = tokenizer.decode(output_id)
        if len(output_id) == num_gen:
            output_ids.append(output_id)
            output_tokens.append(tokens)
            probs.append(output_probs)

    output_ids = torch.tensor(output_ids)
    probs = torch.tensor(probs)
    
    if return_probs:
        return (output_ids, probs)
    else:
        return output_tokens


def distill_on_generated_text(model, tokenizer, in_text, lr=1e-5, num_iter=1, device='cuda', verbose=False):
    #loss_fn = MSELoss()
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=0.01)

    num_outputs = 10

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    with torch.no_grad():
        (gen_ids, probs) = produce_text(model, tokenizer, in_text, num_outputs=num_outputs, device=device, return_probs=True)
        probs = softmax(probs)

    bos_tokens = torch.tensor([tokenizer.bos_token_id]*num_outputs).unsqueeze(1)
    gen_ids = torch.cat((bos_tokens, gen_ids), dim=1)

    model.train()
    for i in range(num_iter):
        # Feed generated outputs through the model with no context
        outputs = model(gen_ids)

        # Extract the logits computed by the model for the generated words
        new_logits = outputs.logits[:,:-1]
        gen_ids_flattened = gen_ids[:,1:].unsqueeze(-1)
        new_logits = new_logits.gather(-1, gen_ids_flattened).squeeze()

        loss = loss_fn(new_logits, probs)

        if verbose:
            print(f"Loss after step {i} of optimization: {loss.item()}")
        loss.backward()
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()

def distill_to_mem_token(model, tokenizer, in_text, lr=1e-5, num_iter=1, device='cuda', verbose=False):
    #loss_fn = MSELoss()
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=0.01)

    num_outputs = 10
    mem_token = '[MEM]'

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_tokens(mem_token)
    model.resize_token_embeddings(len(tokenizer))
    mem_token_id = tokenizer.convert_tokens_to_ids(mem_token)
    with torch.no_grad():
        (gen_ids, probs) = produce_text(model, tokenizer, in_text, num_outputs=num_outputs, device=device, return_probs=True)
        probs = softmax(probs)
    
    bos_tokens = torch.tensor([tokenizer.bos_token_id]*num_outputs).unsqueeze(1)
    mem_tokens = torch.tensor([mem_token_id]*num_outputs).unsqueeze(1)
    gen_ids = torch.cat((bos_tokens, mem_tokens, gen_ids), dim=1)

    model.train()
    embeddings = model.get_input_embeddings()
    embeddings.weight[mem_token_id].retain_grad()
    for i in range(num_iter):
        # Feed generated outputs through the model with no context
        outputs = model(gen_ids)

        # Extract the logits computed by the model for the generated words
        new_logits = outputs.logits[:,1:-1]
        gen_ids_flattened = gen_ids[:,2:].unsqueeze(-1)
        new_logits = new_logits.gather(-1, gen_ids_flattened).squeeze()

        loss = loss_fn(new_logits, probs)

        if verbose:
            print(f"Loss after step {i} of optimization: {loss.item()}")
        loss.backward()

        for i in range(len(tokenizer)):
            if i != mem_token_id:
                embed_size = embeddings.weight.grad[i,:].shape
                embeddings.weight.grad[i,:] = torch.zeros(embed_size)

        for name, param in model.named_parameters():
            if (name != 'model.embed_tokens.weight') and (param.grad is not None):
                param.grad.zero_()

        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()
