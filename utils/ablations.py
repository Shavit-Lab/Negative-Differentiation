import torch
import torch.nn as nn
from typing import List, Dict, Callable
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM, OPTForCausalLM, GPT2Tokenizer
import numpy as np

def return_neuron_indices(
    all_WDs: dict,
    model_type: str,
    target_layers: List[int],
    percent_to_remove: float,
    location: str,
) ->List[np.ndarray]:
    """
    Finds indices of neurons to zero out based on the WDs
    
    Args:
        all_WDs: Dictionary of WDs
        model_type: Type of model, either "pythia" or other
        target_layers: List of layer indices to modify
        percent_to_remove: Percentage of neurons to remove, between 0 and 100
        location: Location of the neurons to remove, either "top", "bottom", or "random"
    
    Returns:
        List of indices of neurons to zero out, one for each layer in target_layers
    """
    # Assert that percent to remove is between 0 and 100
    assert 0 <= percent_to_remove <= 100, "Percent to remove must be between 0 and 100"
    # Assert that location must be a string that is either "top", "bottom", or "random"
    assert location in ["top", "bottom", "random"], "Location must be either 'top', 'bottom', or 'random'"
    WDs_indices = []
    for targeted_layer in target_layers:
        if model_type.lower() == "pythia":
            key = targeted_layer
        elif model_type.lower() == "pythia-6.9b":
            key = f"{targeted_layer}_dense_h_to_4h"
        else:
            key = f"{targeted_layer}_gate_proj"
        if location == "top":
            top_percent = np.percentile(all_WDs[key], 100 - percent_to_remove)
            top_percent_indices = np.where(all_WDs[key] > top_percent)[0]
            WDs_indices.append(top_percent_indices)
        elif location == "bottom":
            bottom_percent = np.percentile(all_WDs[key], percent_to_remove)
            bottom_percent_indices = np.where(all_WDs[key] < bottom_percent)[0]
            WDs_indices.append(bottom_percent_indices)
        elif location == "random":
            top_percent = np.percentile(all_WDs[key], 100 - percent_to_remove)
            top_percent_indices = np.where(all_WDs[key] > top_percent)[0]
            num_elements = len(top_percent_indices)
            random_indices = np.random.choice(range(all_WDs[key].shape[0]), size=num_elements, replace=False)
            WDs_indices.append(random_indices)
    return WDs_indices
                

def create_activation_modifier(
    model: nn.Module,
    target_layers: List[int],
    target_projs: List[str],
    modification_fn: Callable[[torch.Tensor], torch.Tensor]
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Creates hooks to modify activations during model inference.
    
    Args:
        model: The model to modify
        target_layers: List of layer indices to modify
        target_projs: List of projection names to modify (e.g. ["gate_proj", "up_proj"])
        modification_fn: Function that takes a tensor and returns the modified tensor
    
    Returns:
        List of hooks that can be removed later
    """
    hooks = []
    
    def get_modification_hook(layer_idx: int, proj_name: str):
        def hook(module, input, output):
            # Apply the modification function to the output
            modified_output = modification_fn(output)
            return modified_output
        return hook
    
    # Register hooks based on model architecture
    if type(model).__name__ == "GPTNeoXForCausalLM":
        for layer_idx in target_layers:
            for proj_name in target_projs:
                module = model.gpt_neox.layers[layer_idx].mlp.__getattr__(proj_name)
                hook = module.register_forward_hook(get_modification_hook(layer_idx, proj_name))
                hooks.append(hook)
                print(f"Hook registered for layer {layer_idx} and projection {proj_name}")
    elif type(model).__name__ == "OPTForCausalLM":
        for layer_idx in target_layers:
            for proj_name in target_projs:
                module = model.model.decoder.layers[layer_idx].__getattr__(proj_name)
                hook = module.register_forward_hook(get_modification_hook(layer_idx, proj_name))
                hooks.append(hook)
    else:  # For LLaMA and similar architectures
        for layer_idx in target_layers:
            for proj_name in target_projs:
                module = model.model.layers[layer_idx].mlp.__getattr__(proj_name)
                hook = module.register_forward_hook(get_modification_hook(layer_idx, proj_name))
                hooks.append(hook)
    
    return hooks

def zero_out_neurons(neuron_indices: List[int]) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates a function that zeros out specific neurons in the activation tensor.
    
    Args:
        neuron_indices: List of neuron indices to zero out
    
    Returns:
        A function that takes a tensor and returns a modified tensor with specified neurons zeroed out
    """
    def modification_fn(tensor: torch.Tensor) -> torch.Tensor:
        # Create a copy to avoid modifying the original tensor
        modified = tensor.clone()
        # Zero out the specified neurons
        modified[..., neuron_indices] = 0
        return modified
    return modification_fn

def zero_out_negative_activations(neuron_indices: List[int]) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates a function that zeros out negative activations for neurons in the given indices.
    
    Args:
        neuron_indices: List of neuron indices to zero out
    
    Returns:
        A function that takes a tensor and returns a modified tensor with specified neurons zeroed out
    """
    def modification_fn(tensor: torch.Tensor) -> torch.Tensor:
        # Create a copy to avoid modifying the original tensor
        modified = tensor.clone()
        # For the specified neurons, set the negative values to zero
        modified[..., neuron_indices] = torch.clamp(modified[..., neuron_indices], min=0)
        return modified
    return modification_fn

def zero_out_positive_activations(neuron_indices: List[int]) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates a function that zeros out negative activations for neurons in the given indices.
    
    Args:
        neuron_indices: List of neuron indices to zero out
    
    Returns:
        A function that takes a tensor and returns a modified tensor with specified neurons zeroed out
    """
    def modification_fn(tensor: torch.Tensor) -> torch.Tensor:
        # Create a copy to avoid modifying the original tensor
        modified = tensor.clone()
        # For the specified neurons, set the negative values to zero
        modified[..., neuron_indices] = torch.clamp(modified[..., neuron_indices], max=0)
        return modified
    return modification_fn

def load_model(model_path):

    # Load model and tokenizer
    print(f"Loading model from {model_path}")
    model_name = model_path

    if "llama" in model_name.lower():
        print("Using Llama model")
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Load model
        stored_model = AutoModelForCausalLM.from_pretrained(model_name)
        model = stored_model

    if "qwen" in model_name.lower():
        print("Using Qwen model")
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Load model
        stored_model = AutoModelForCausalLM.from_pretrained(model_name)
        model = stored_model

    if "mistral" in model_name.lower():
        print("Using Mistral model")
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Load model
        stored_model = AutoModelForCausalLM.from_pretrained(model_name)
        model = stored_model

    elif 'pythia' in model_name.lower():
        print("Using Pythia model")
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Load model
        stored_model = GPTNeoXForCausalLM.from_pretrained(model_name)
        model = stored_model

    elif 'opt' in model_name.lower():
        print("Using OPT model")
        # Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Load model
        stored_model = OPTForCausalLM.from_pretrained(model_name)
        model = stored_model

    return model, tokenizer

def return_modified_model(model, target_layers, target_projs, neuron_indices, ablation_type="negative"):
    print(f"Applying activation modifications in {ablation_type} pre-activation")
    if len(target_layers) != len(neuron_indices):
        raise ValueError(f"Number of target layers ({len(target_layers)}) and neuron indices ({len(neuron_indices)}) must be the same")
    hooks = []
    for i in range(len(target_layers)):
        if ablation_type == "negative":
            zero_fn = zero_out_negative_activations(neuron_indices[i])
        elif ablation_type == "positive":
            zero_fn = zero_out_positive_activations(neuron_indices[i])
        elif ablation_type == "both":
            zero_fn = zero_out_neurons(neuron_indices[i])
        else:
            raise ValueError(f"Invalid ablation type: {ablation_type}")
        curr_hooks = create_activation_modifier(
            model = model,
            target_layers = [target_layers[i]],
            target_projs = target_projs,
            modification_fn = zero_fn
        )
        hooks.extend(curr_hooks)
    return model, hooks

def return_modified_model_from_path(model_path, target_layers, target_projs, neuron_indices, ablation_type="negative"):

    model, tokenizer = load_model(model_path)
    return return_modified_model(model, target_layers, target_projs, neuron_indices, ablation_type)

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Sequence

# ------------------------------
# Core activation function modifiers
# ------------------------------
class SelectivePosReLU(nn.Module):
    """
    For a selected subset of features (neurons), use ReLU on the positive side
    and keep the base activation (SiLU or GELU) on the negative side.
    For all other features, return the base activation unchanged.

    f(x) for selected dims:   x if x>0 else base(x)
    f(x) for non-selected:    base(x)
    """
    def __init__(
        self,
        base: str = "silu",              # "silu" or "gelu"
        idxs: Optional[Sequence[int]] = None,  # intermediate-dim indices to modify
        dim_size: Optional[int] = None,  # intermediate dimension, if known at init
        gelu_approx: str = "none"        # "none" or "tanh" (PyTorch approximate GELU)
    ):
        super().__init__()
        base = base.lower()
        if base not in ("silu", "gelu"):
            raise ValueError("base must be 'silu' or 'gelu'")

        self.base_name = base
        if base == "silu":
            self.base_act = nn.SiLU()
        else:
            # exact by default; set approximate="tanh" for GPT-NeoX-style fast gelu
            self.base_act = nn.GELU(approximate=gelu_approx)

        # store indices as a buffer mask; create lazily if dim_size isn't known yet
        self.register_buffer("idx_mask", None, persistent=False)
        if idxs is not None and dim_size is not None:
            mask = torch.zeros(dim_size, dtype=torch.bool)
            mask[list(idxs)] = True
            self.idx_mask = mask

        self._pending_idxs = list(idxs) if idxs is not None else None

    def _lazy_init_mask(self, x_last_dim: int):
        if self.idx_mask is None:
            if self._pending_idxs is None:
                # If no subset was specified, default to "all" (apply the mixed ReLU everywhere)
                mask = torch.ones(x_last_dim, dtype=torch.bool, device=self.base_act.parameters().__next__().device if any(True for _ in self.base_act.parameters()) else None)
            else:
                mask = torch.zeros(x_last_dim, dtype=torch.bool)
                mask[self._pending_idxs] = True
            self.idx_mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [..., D] where D is the intermediate (4h) dimension.
        """
        base_out = self.base_act(x)
        # Ensure mask exists and is on the correct device/dtype
        self._lazy_init_mask(x.size(-1))
        mask = self.idx_mask
        if mask.device != x.device:
            mask = mask.to(x.device)

        # Broadcast mask to x shape on the last dim
        for _ in range(x.dim() - 1):
            mask = mask.unsqueeze(0)

        # Mixed output on selected dims; base elsewhere
        relu_side = torch.where(x > 0, x, base_out)
        return torch.where(mask, relu_side, base_out)

class SelectiveSignFlip(nn.Module):
    """
    For a selected subset of features (neurons), flip the sign of the negative values after the base activation (SiLU or GELU).
    For all other features, return the base activation unchanged.

    f(x) for selected dims:   base(x) if base(x) > 0 else -base(x)
    f(x) for non-selected:    base(x)
    """
    def __init__(
        self,
        base: str = "silu",              # "silu" or "gelu"
        idxs: Optional[Sequence[int]] = None,  # intermediate-dim indices to modify
        dim_size: Optional[int] = None,  # intermediate dimension, if known at init
        gelu_approx: str = "none"        # "none" or "tanh" (PyTorch approximate GELU)
    ):
        super().__init__()
        base = base.lower()
        if base not in ("silu", "gelu"):
            raise ValueError("base must be 'silu' or 'gelu'")

        self.base_name = base
        if base == "silu":
            self.base_act = nn.SiLU()
        else:
            # exact by default; set approximate="tanh" for GPT-NeoX-style fast gelu
            self.base_act = nn.GELU(approximate=gelu_approx)

        # store indices as a buffer mask; create lazily if dim_size isn't known yet
        self.register_buffer("idx_mask", None, persistent=False)
        if idxs is not None and dim_size is not None:
            mask = torch.zeros(dim_size, dtype=torch.bool)
            mask[list(idxs)] = True
            self.idx_mask = mask

        self._pending_idxs = list(idxs) if idxs is not None else None

    def _lazy_init_mask(self, x_last_dim: int):
        if self.idx_mask is None:
            if self._pending_idxs is None:
                # If no subset was specified, default to "all" (apply the mixed ReLU everywhere)
                mask = torch.ones(x_last_dim, dtype=torch.bool, device=self.base_act.parameters().__next__().device if any(True for _ in self.base_act.parameters()) else None)
            else:
                mask = torch.zeros(x_last_dim, dtype=torch.bool)
                mask[self._pending_idxs] = True
            self.idx_mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [..., D] where D is the intermediate (4h) dimension.
        """
        base_out = self.base_act(x)
        # Ensure mask exists and is on the correct device/dtype
        self._lazy_init_mask(x.size(-1))
        mask = self.idx_mask
        if mask.device != x.device:
            mask = mask.to(x.device)

        # Broadcast mask to x shape on the last dim
        for _ in range(x.dim() - 1):
            mask = mask.unsqueeze(0)

        # Sign flip on negative values of selected dims, otherwise same
        sign_flip_side = torch.where(base_out < 0, -base_out, base_out)
        return torch.where(mask, sign_flip_side, base_out)


# ------------------------------
# Utilities to choose neuron indices
# ------------------------------
def choose_indices(dim: int, idxs: Optional[Sequence[int]] = None, frac: Optional[float] = None, seed: int = 0) -> List[int]:
    if idxs is not None:
        # validate indices
        bad = [i for i in idxs if i < 0 or i >= dim]
        if bad:
            raise ValueError(f"Indices out of range for dim={dim}: {bad}")
        return list(idxs)

    if frac is None:
        raise ValueError("Provide either explicit idxs or a frac in (0,1].")

    if not (0 < frac <= 1.0):
        raise ValueError("frac must be in (0,1].")

    k = max(1, int(round(frac * dim)))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(dim, generator=g)
    return perm[:k].tolist()


# ------------------------------
# Patchers for LLaMA and Pythia/GPT-NeoX
# ------------------------------
def patch_llama(
    model,
    type,
    idxs_per_layer: Optional[Dict[int, Sequence[int]]] = None,
    frac: Optional[float] = None,
    seed: int = 0,
):
    """
    Replace the LLaMA MLP gate activation with SelectivePosReLU or SelectiveSignFlip on a subset of neurons.
    Target: layer.mlp.act_fn (applied to gate_proj output in SwiGLU).
    """
    # Grab the decoder layers in HF LLaMA
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        raise RuntimeError("Could not locate LLaMA layers at model.model.layers")

    assert type in ["posrelu", "signflip"], "Type must be 'posrelu' or 'signflip'"

    for i, layer in enumerate(layers):
        mlp = layer.mlp
        inter_dim = mlp.gate_proj.out_features

        # stash original to allow restore later
        if not hasattr(mlp, "_orig_act_fn"):
            mlp._orig_act_fn = mlp.act_fn

        if idxs_per_layer is not None and i in idxs_per_layer:
            idxs = idxs_per_layer[i]
        else:
            idxs = choose_indices(inter_dim, idxs=None, frac=frac, seed=seed + i)
        if type == "posrelu":
            mlp.act_fn = SelectivePosReLU(base="silu", idxs=idxs, dim_size=inter_dim)
        elif type == "signflip":
            mlp.act_fn = SelectiveSignFlip(base="silu", idxs=idxs, dim_size=inter_dim)
        else:
            raise ValueError(f"Invalid type: {type}")


def patch_pythia(
    model,
    type,
    idxs_per_layer: Optional[Dict[int, Sequence[int]]] = None,
    frac: Optional[float] = None,
    seed: int = 0,
    gelu_approx: str = "none",  # set "tanh" to mimic fast gelu if you want
):
    """
    Replace the Pythia/GPT-NeoX MLP activation with SelectivePosReLU or SelectiveSignFlip on a subset
    of the dense_h_to_4h neurons.
    Target: layer.mlp.act (the activation after dense_h_to_4h).
    """
    # GPTNeoX layers live at model.gpt_neox.layers
    gpt_neox = getattr(model, "gpt_neox", None)
    layers = getattr(gpt_neox, "layers", None)
    if layers is None:
        raise RuntimeError("Could not locate GPT-NeoX layers at model.gpt_neox.layers")

    assert type in ["posrelu", "signflip"], "Type must be 'posrelu' or 'signflip'"

    for i, layer in enumerate(layers):
        mlp = layer.mlp
        inter_dim = mlp.dense_h_to_4h.out_features

        if not hasattr(mlp, "_orig_act"):
            mlp._orig_act = mlp.act

        if idxs_per_layer is not None and i in idxs_per_layer:
            idxs = idxs_per_layer[i]
        else:
            idxs = choose_indices(inter_dim, idxs=None, frac=frac, seed=seed + i)

        if type == "posrelu":
            mlp.act = SelectivePosReLU(base="gelu", idxs=idxs, dim_size=inter_dim, gelu_approx=gelu_approx)
        elif type == "signflip":
            mlp.act = SelectiveSignFlip(base="gelu", idxs=idxs, dim_size=inter_dim, gelu_approx=gelu_approx)
        else:
            raise ValueError(f"Invalid type: {type}")

# ------------------------------
# Restore helpers
# ------------------------------
def restore_llama_activation(model):
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        return
    for layer in layers:
        mlp = layer.mlp
        if hasattr(mlp, "_orig_act_fn"):
            mlp.act_fn = mlp._orig_act_fn

def restore_pythia_activation(model):
    gpt_neox = getattr(model, "gpt_neox", None)
    layers = getattr(gpt_neox, "layers", None)
    if layers is None:
        return
    for layer in layers:
        mlp = layer.mlp
        if hasattr(mlp, "_orig_act"):
            mlp.act = mlp._orig_act