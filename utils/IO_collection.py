from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def collect_input_ids(model, dataset_name, tokenizer, num_batches=256, sequence_length=2048, batch_size=1, target_layers=[1, 5, 10], target_projs=["gate_proj"]):
    if dataset_name == "wikitext2-test":
        # Load WikiText-2 dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        
        # Tokenize the entire validation set as one long sequence
        full_text = "\n\n".join(dataset['test']['text'])

        tokenized_text = tokenizer(full_text, return_tensors='pt')
        print("Using wikitext2-test dataset")
        # Create sliding windows
        samples = []
        all_input_ids = []
        total_length = tokenized_text.input_ids.shape[1]
        
        for i in range(0, total_length//sequence_length):
            start = i * sequence_length
            end = start + sequence_length
            if end > total_length:
                print(f"end > total_length: {end} > {total_length}")
                break
            
            input_ids = tokenized_text.input_ids[:, start:end]
            samples.append({
                'input_ids': input_ids.squeeze(),
                'attention_mask': torch.ones_like(input_ids.squeeze()),
            })
            all_input_ids.append(input_ids.squeeze())

            if len(samples) >= num_batches * batch_size:
                print(f"len(samples) >= num_batches * batch_size: {len(samples)} >= {num_batches} * {batch_size}")
                break

        # Create dataset from samples
        class WikiTextDataset(torch.utils.data.Dataset):
            def __init__(self, samples):
                self.samples = samples
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                return self.samples[idx]
        
        # Create dataloader
        dataset = WikiTextDataset(samples)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
        return all_input_ids
    return None

def capture_mlp_IO(model, input_ids, target_layers=[1, 5, 10], target_projs=["gate_proj"]):
    """Capture MLP activations for specified layers during a forward pass"""
    linear_inputs = {}
    linear_outputs = {}
    
    # Define hooks to capture activations
    def get_IO_hook(layer_idx, proj_name):
        def hook(module, input, output):
            linear_inputs[f"layer_{layer_idx}_{proj_name}"] = input[0].detach().cpu()
            linear_outputs[f"layer_{layer_idx}_{proj_name}"] = output.detach().cpu()
        return hook
    
    # Register hooks for the target layers
    hooks = []
    if type(model).__name__ == "GPTNeoXForCausalLM":
        for layer_idx in target_layers:
            for proj_name in target_projs:
                module = model.gpt_neox.layers[layer_idx].mlp.__getattr__(proj_name)
                hook = module.register_forward_hook(get_IO_hook(layer_idx, proj_name))
                hooks.append(hook)
    elif type(model).__name__ == "OPTForCausalLM":
        for layer_idx in target_layers:
            for proj_name in target_projs:
                module = model.model.decoder.layers[layer_idx].__getattr__(proj_name)
                hook = module.register_forward_hook(get_IO_hook(layer_idx, proj_name))
                hooks.append(hook)
    else:
        for layer_idx in target_layers:
            for proj_name in target_projs:
                module = model.model.layers[layer_idx].mlp.__getattr__(proj_name)
                hook = module.register_forward_hook(get_IO_hook(layer_idx, proj_name))
                hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # return activations
    return linear_inputs, linear_outputs

def collect_IO_from_dataset(model, dataset_name, tokenizer, num_batches=256, sequence_length=2048, batch_size=1, target_layers=[1, 5, 10], target_projs=["gate_proj"]):
    device = model.device
    """Collect input and outputs (pre-activation)"""

    if dataset_name == "wikitext2":
        # Load WikiText-2 dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        
        # Tokenize the entire validation set as one long sequence
        full_text = " ".join(dataset['train']['text'])
        tokenized_text = tokenizer(full_text, return_tensors='pt')
        
        # Create sliding windows
        samples = []
        total_length = tokenized_text.input_ids.shape[1]
        
        # Generate evenly spaced windows to cover the text
        # Use stride to control overlap between windows
        stride = sequence_length // 2  # 50% overlap between windows
        for i in range(0, total_length - sequence_length, stride):
            input_ids = tokenized_text.input_ids[:, i:i + sequence_length]
            samples.append({
                'input_ids': input_ids.squeeze(),
                'attention_mask': torch.ones_like(input_ids.squeeze()),
            })
            
            if len(samples) >= num_batches * batch_size:
                break
        
        # Create dataset from samples
        class WikiTextDataset(torch.utils.data.Dataset):
            def __init__(self, samples):
                self.samples = samples
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                return self.samples[idx]
        
        # Create dataloader
        dataset = WikiTextDataset(samples)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )

    elif dataset_name == "wikitext2-test":
        # Load WikiText-2 dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        
        # Tokenize the entire validation set as one long sequence
        full_text = "\n\n".join(dataset['test']['text'])

        tokenized_text = tokenizer(full_text, return_tensors='pt')
        print("Using wikitext2-test dataset")
        # Create sliding windows
        samples = []
        all_input_ids = []
        total_length = tokenized_text.input_ids.shape[1]
        
        for i in range(0, total_length//sequence_length):
            start = i * sequence_length
            end = start + sequence_length
            if end > total_length:
                print(f"end > total_length: {end} > {total_length}")
                break
            
            input_ids = tokenized_text.input_ids[:, start:end]
            samples.append({
                'input_ids': input_ids.squeeze(),
                'attention_mask': torch.ones_like(input_ids.squeeze()),
            })
            all_input_ids.append(input_ids.squeeze())

            if len(samples) >= num_batches * batch_size:
                print(f"len(samples) >= num_batches * batch_size: {len(samples)} >= {num_batches} * {batch_size}")
                break

        class WikiTextDataset(torch.utils.data.Dataset):
            def __init__(self, samples):
                self.samples = samples
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                return self.samples[idx]
        
        # Create dataloader
        dataset = WikiTextDataset(samples)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Initialize dictionary for collated activations
    collated_inputs = {}
    collated_outputs = {}
    
    # Process batches
    for i, batch in tqdm(enumerate(dataloader), total=num_batches):
        if i >= num_batches:
            break
        
        # Move input to device
        if dataset_name == "wikitext2":
            input_ids = batch['input_ids'].to(device)
        else:  # gsm8k
            input_ids = batch['input_ids'].to(device)
        
        # Capture activations
        batch_inputs, batch_outputs = capture_mlp_IO(model, input_ids, target_layers=target_layers, target_projs=target_projs)
        
        # Initialize collated_activations with empty lists for each key on first batch
        if i == 0:
            for key in batch_inputs.keys():
                collated_inputs[key] = []
            for key in batch_outputs.keys():
                collated_outputs[key] = []
        # Add batch inputs to collated inputs
        for key, value in batch_inputs.items():
            collated_inputs[key].append(value)
        
        # Add batch activations to collated activations
        for key, value in batch_outputs.items():
            collated_outputs[key].append(value)
    
    for key in collated_inputs:
        collated_inputs[key] = torch.cat(collated_inputs[key], dim=0)

    for key in collated_outputs:
        collated_outputs[key] = torch.cat(collated_outputs[key], dim=0)
    
    print(f"Completed collecting activations for {dataset_name}")

    print("Final input shapes:")
    for key, value in collated_inputs.items():
        print(f"{key}: {value.shape}")

    print("Final activation shapes:")
    for key, value in collated_outputs.items():
        print(f"{key}: {value.shape}")
    
    return collated_inputs, collated_outputs