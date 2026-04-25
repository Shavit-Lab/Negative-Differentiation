import json
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Dict

from tqdm import tqdm

import torch

def read_jsonl_pairs(template_dir: Path) -> List[Tuple[str, str, str]]:
    """
    Read minimal pairs from *.jsonl files in template_dir. Each line must have:
      - sentence_good: grammatical sentence
      - sentence_bad:  ungrammatical sentence
    Returns list of (good, bad, source_file).
    """
    pairs = []
    for fp in sorted(template_dir.glob("*.jsonl")):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                good = ex.get("sentence_good")
                bad = ex.get("sentence_bad")
                if good and bad:
                    pairs.append((good, bad, fp.name))
    if not pairs:
        raise ValueError(f"No JSONL pairs found under: {template_dir}")
    return pairs


def read_lm_syneval_pairs(template_dir: Path) -> List[Tuple[str, str, str]]:
    """
    Minimal stub in case you're using the Marvin & Linzen generator output.
    Their generator writes per-phenomenon files (often TSV-like or txt).
    If you use that route, adjust this loader to match the file format it produced.
    For now we raise if no JSONL is found; recommend using the prebuilt JSONL (Refining-TSE).
    """
    # If you want to support the LM_syneval outputs directly, inspect the files created by
    # `make_templates.py` and parse them into (good, bad, source).
    raise NotImplementedError(
        "This script expects JSONL files (use refining-tse/data/ML). "
        "If you generated files via LM_syneval/make_templates.py, "
        "adapt `read_lm_syneval_pairs` to that output format."
    )


@torch.no_grad()
def sentence_logprob_avg(
    model, tokenizer, text: str, device: torch.device
) -> float:
    """
    Compute average log-prob per token for a sentence under a causal LM.
    We predict token t_i from prefix t_<1..i-1>, so we shift and
    ignore the first position (no previous token to predict).
    """
    # Tokenize with BOS if available; set pad_token to eos if missing for padding compatibility
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attn_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)

    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
    # logits: [1, T, V]
    logits = outputs.logits[:, :-1, :]            # predict next token
    targets = input_ids[:, 1:]                    # next tokens
    mask = attn_mask[:, 1:].bool()                # ignore padding and the first pos

    log_probs = torch.log_softmax(logits, dim=-1)
    tgt_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
    if mask is not None:
        tgt_log_probs = tgt_log_probs.masked_select(mask)

    # average per-token log prob
    return tgt_log_probs.mean().item()


def batched_scores(
    model, tokenizer, texts: List[str], device: torch.device, batch_size: int = 8
) -> List[float]:
    scores = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        # batch encode
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attn_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits  # [B,T,V]
            logits = logits[:, :-1, :]
            targets = input_ids[:, 1:]
            mask = attn_mask[:, 1:].bool()

            log_probs = torch.log_softmax(logits, dim=-1)
            tgt_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
            # mask padding + first position
            tgt_log_probs = tgt_log_probs.masked_fill(~mask, 0.0)

            # average per example: sum / (#valid tokens)
            lengths = mask.sum(dim=1).clamp(min=1).to(tgt_log_probs.dtype)
            avg = (tgt_log_probs.sum(dim=1) / lengths).tolist()
            scores.extend(avg)
    return scores


def evaluate_pairs(
    model, tokenizer, pairs: List[Tuple[str, str, str]], device: torch.device, batch_size: int = 8
) -> Dict:
    """
    Returns:
      {
        "overall_acc": float,
        "per_file": {filename: accuracy_float},
        "n": int
      }
    """
    all_good = [g for g, b, s in pairs]
    all_bad  = [b for g, b, s in pairs]
    all_src  = [s for g, b, s in pairs]

    good_scores = batched_scores(model, tokenizer, all_good, device, batch_size)
    bad_scores  = batched_scores(model, tokenizer, all_bad,  device, batch_size)

    correct = [1 if g > b else 0 for g, b in zip(good_scores, bad_scores)]
    overall = sum(correct) / len(correct)

    # per-template-file accuracy
    per_file: Dict[str, List[int]] = {}
    for ok, src in zip(correct, all_src):
        per_file.setdefault(src, []).append(ok)
    per_file_acc = {k: sum(v) / len(v) for k, v in per_file.items()}

    return {
        "overall_acc": overall,
        "per_file": per_file_acc,
        "n": len(pairs),
    }

def _clean_text(s: str) -> str:
    # Normalize whitespace, strip stray spaces
    return re.sub(r"\s+", " ", s).strip()

def read_pickle_pairs(template_dir: Path) -> List[Tuple[str, str, str]]:
    """
    Read minimal pairs from *.pickle files produced by LM_syneval/make_templates.py.
    Each pickle is a dict: {case_name: [(good, bad) ...] or [(good, intrusive, bad) ...]}
    Returns list of (good, bad, source_file).
    """
    pairs: List[Tuple[str, str, str]] = []
    for fp in sorted(template_dir.glob("*.pickle")):
        with fp.open("rb") as f:
            data = pickle.load(f)

        # data: dict[case_name -> list[tuple]]
        for case_name, tuples in data.items():
            for tup in tuples:
                # agreement: (good, bad)
                # NPI: (good, intrusive, bad)
                if not isinstance(tup, (list, tuple)) or len(tup) not in (2, 3):
                    continue
                good = tup[0]
                bad  = tup[-1]  # last item is ungrammatical in both 2- and 3-tuples

                # Filter out generator placeholders, sanitize whitespace
                if not good or not bad or "None" in (good, bad):
                    continue
                good = _clean_text(good)
                bad  = _clean_text(bad)
                if good and bad and good != "None" and bad != "None":
                    # Include both filename and case for traceability
                    src = f"{fp.name}:{case_name}"
                    pairs.append((good, bad, src))

    if not pairs:
        raise ValueError(f"No usable pairs found in {template_dir} (looked for *.pickle)")
    return pairs

def evaluate_tse(model, tokenizer, device, batch_size=128, templates_path="/home/linghao/LM_syneval/templates_out"):
    """
    Evaluate TSE benchmark on a model.
    
    Args:
        model: The language model to evaluate
        tokenizer: The tokenizer for the model
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        templates_path: Path to the templates directory
    """
    templates_dir = Path(templates_path)
    pairs = read_pickle_pairs(templates_dir)
    return evaluate_pairs(model, tokenizer, pairs, device, batch_size)

def get_pair_counts(templates_path = "/home/linghao/LM_syneval/templates_out"):
    templates_dir = Path(templates_path)
    pairs = read_pickle_pairs(templates_dir)
    return pairs
