import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import math
from typing import List, Optional, Dict, Tuple

def load_wikitext2_dataset(tokenizer, sequence_length: int = 2048) -> List[torch.Tensor]:
    """Load and prepare WikiText-2 dataset for perplexity evaluation."""
    # Load WikiText-2 dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    # Tokenize the entire validation set as one long sequence
    full_text = "\n\n".join(dataset['validation']['text'])
    tokenized_text = tokenizer(full_text, return_tensors='pt')
    
    # Create non-overlapping sequences
    sequences = []
    total_length = tokenized_text.input_ids.shape[1]
    
    for i in range(0, total_length, sequence_length):
        if i + sequence_length > total_length:
            break
        sequences.append(tokenized_text.input_ids[:, i:i + sequence_length])
    
    return sequences

@torch.no_grad()
def calculate_perplexity(
    model: nn.Module,
    sequences: List[torch.Tensor],
    device: torch.device,
    batch_size: int = 1
) -> float:
    """Calculate perplexity on the given sequences."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for sequence in tqdm(sequences, desc="Calculating perplexity"):
            sequence = sequence.to(device)
            outputs = model(sequence, labels=sequence)
            loss = outputs.loss
            total_loss += loss.item() * sequence.numel()
            total_tokens += sequence.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

@torch.no_grad()
def calculate_per_token_perplexity(
    model: nn.Module,
    sequences: List[torch.Tensor],         # list of 1D int64 tensors (token ids)
    device: torch.device,
    batch_size: int = 1,
    pad_token_id: Optional[int] = None,    # required if sequences have different lengths
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute per-token NLL and Perplexity for each sequence.

    Returns:
        List of tuples (nll_per_token, ppl_per_token) for each input sequence.
        Each is a float tensor of shape [seq_len].
        Positions that aren't evaluated (e.g., first token, padding) are NaN.

    Notes:
      - For causal LMs, token t is predicted from tokens < t.
      - We shift logits left: logits[:, :-1, :] vs labels[:, 1:].
      - The first position (index 0) is always NaN, because there's no previous context to predict it.
    """
    model.eval()

    # Helper to batch a slice of sequences with optional padding
    def make_batch(seq_slice: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        lengths = [len(s) for s in seq_slice]
        max_len = max(lengths)
        if pad_token_id is None and any(L != max_len for L in lengths):
            raise ValueError(
                "Variable-length sequences detected. Provide pad_token_id so we can batch with padding."
            )

        if pad_token_id is None:
            # All same length — just stack and full mask
            input_ids = torch.stack(seq_slice, dim=0)
            attn_mask = torch.ones_like(input_ids, dtype=torch.long)
        else:
            # Pad to max_len
            input_ids = torch.full((len(seq_slice), max_len), pad_token_id, dtype=torch.long)
            attn_mask = torch.zeros((len(seq_slice), max_len), dtype=torch.long)
            for i, s in enumerate(seq_slice):
                L = len(s)
                input_ids[i, :L] = s
                attn_mask[i, :L] = 1

        return input_ids.to(device), attn_mask.to(device), lengths

    results: List[Tuple[torch.Tensor, torch.Tensor]] = []

    # We'll accumulate results in order; process in mini-batches
    for b_start in tqdm(range(0, len(sequences), batch_size)):
        batch_seqs = sequences[b_start:b_start + batch_size]
        input_ids, attention_mask, lengths = make_batch(batch_seqs)

        # Forward
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, L, V]

        # Shift for causal LM: predict token t using positions up to t-1
        shift_logits = logits[:, :-1, :].contiguous()              # [B, L-1, V]
        shift_labels = input_ids[:, 1:].contiguous()               # [B, L-1]
        shift_mask   = attention_mask[:, 1:].contiguous()          # [B, L-1]

        # Cross-entropy per token (no reduction)
        V = shift_logits.size(-1)
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            reduction='none'
        ).view(shift_labels.size())                                 # [B, L-1]

        # Mask out padded positions (set to NaN so they don't affect means later)
        per_token_loss = per_token_loss.masked_fill(shift_mask == 0, float('nan'))

        # Re-align to the original length L by prepending a NaN column for position 0
        nan_col = torch.full((per_token_loss.size(0), 1), float('nan'), device=per_token_loss.device)
        per_token_loss_full = torch.cat([nan_col, per_token_loss], dim=1)  # [B, L]

        # Convert to perplexity per token
        per_token_ppl = torch.exp(per_token_loss_full)

        # Now slice back to each original sequence length and move to CPU
        for i, L in enumerate(lengths):
            nll_i = per_token_loss_full[i, :L].detach().cpu()
            ppl_i = per_token_ppl[i, :L].detach().cpu()
            results.append((nll_i, ppl_i))

    return results