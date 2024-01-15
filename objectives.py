from torch.optim import SGD
from torch.nn import CrossEntropyLoss, KLDivLoss, MSELoss
import torch

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