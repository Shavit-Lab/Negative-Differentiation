from dataclasses import dataclass
from typing import List, Dict

import torch
import gc
from datasets import load_dataset, get_dataset_config_names
from tqdm import tqdm


@dataclass
class PairBatch:
    good: List[str]
    bad: List[str]


def compute_logprobs(model, tokenizer, sentences: List[str], device: torch.device, batch_size: int = 16) -> List[float]:
    """
    Compute total (unnormalized) log-probabilities of each sentence under an autoregressive LM.
    Sum of log p(x_t | x_<t). No length normalization (BLiMP pairs are near-equal length).
    """
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
            ).to(device)

            # For causal LMs, we score next-token likelihoods by shifting.
            # logits: [B, T, V]; labels: [B, T]
            outputs = model(**enc)
            logits = outputs.logits  # [B, T, V]
            log_probs = torch.log_softmax(logits, dim=-1)

            input_ids = enc["input_ids"]  # [B, T]
            attn = enc.get("attention_mask", torch.ones_like(input_ids))

            # Shift for next-token prediction
            # labels are the next tokens; we ignore the first position
            shifted_input_ids = input_ids[:, 1:].contiguous()        # [B, T-1]
            shifted_attn = attn[:, 1:].contiguous()                   # [B, T-1]
            shifted_log_probs = log_probs[:, :-1, :].contiguous()     # [B, T-1, V]

            # Gather log-prob of the actually observed next token at each position
            token_logp = torch.gather(shifted_log_probs, dim=-1, index=shifted_input_ids.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

            # Mask out padding positions
            token_logp = token_logp * shifted_attn

            # Sum over time to get total sentence log-prob
            sent_logp = token_logp.sum(dim=-1)  # [B]
            out.extend(sent_logp.tolist())
    gc.collect()
    torch.cuda.empty_cache()
    return out


def eval_pairs(model, tokenizer, pairs: PairBatch, device: torch.device, batch_size: int = 16) -> Dict[str, float]:
    """
    Return accuracy on a set of minimal pairs:
    accuracy = mean( logp(good) > logp(bad) ).
    """
    all_sents = pairs.good + pairs.bad
    all_logps = compute_logprobs(model, tokenizer, all_sents, device, batch_size=batch_size)

    # garbage collect


    n = len(pairs.good)
    good_logp = all_logps[:n]
    bad_logp = all_logps[n:]
    correct = sum(1 for g, b in zip(good_logp, bad_logp) if g > b)

    return {
        "accuracy": correct / n if n else 0.0,
        "num_pairs": n,
        "sum_correct": correct,
        "good_logp": good_logp,
        "bad_logp": bad_logp,
    }


def eval_blimp(model, tokenizer, device, batch_size=128, debug=False):
    print("Fetching BLiMP configs...")
    configs = get_dataset_config_names("nyu-mll/blimp")
    print(f"Evaluating {len(configs)} BLiMP sub-benchmarks...")

    overall_correct = 0
    overall_total = 0
    per_config_scores = []
    per_phenomenon_scores = {}
    
    for cfg in tqdm(configs):
        ds = load_dataset("nyu-mll/blimp", cfg, split="train")  # BLiMP is delivered as a single split
        # Expected fields: sentence_good, sentence_bad, phenomenon, UID
        goods = [ex["sentence_good"] for ex in ds]
        bads  = [ex["sentence_bad"] for ex in ds]
        phenomenon = ds['linguistics_term'][0]
        if phenomenon not in per_phenomenon_scores:
            per_phenomenon_scores[phenomenon] = []
        pairs = PairBatch(good=goods, bad=bads)

        res = eval_pairs(model, tokenizer, pairs, device, batch_size=batch_size)
        acc = res["accuracy"]
        if debug:
            diffs = [g - b for g,b in zip(res["good_logp"], res["bad_logp"])]
            median_diff = float(torch.median(torch.tensor(diffs)))
            mean_diff = float(torch.mean(torch.tensor(diffs)))
            print(f"{cfg}: acc={acc*100:.2f}%, median_diff={median_diff:.3f}, mean_diff={mean_diff:.3f}")
            # print(f"{cfg}: {acc * 100:.2f}%")
        per_phenomenon_scores[phenomenon].append((cfg, acc, res["num_pairs"], res["good_logp"], res["bad_logp"]))
        per_config_scores.append((cfg, acc, res["num_pairs"], res["good_logp"], res["bad_logp"]))
        overall_correct += res["sum_correct"]
        overall_total += res["num_pairs"]

    print(f"Overall accuracy: {overall_correct / overall_total * 100:.2f}%  ({overall_correct}/{overall_total})")
    print("Per phenomenon accuracy:")
    for phenomenon, scores in per_phenomenon_scores.items():
        print(f"{phenomenon}: {sum(score[1] * score[2] for score in scores) / sum(score[2] for score in scores) * 100:.2f}%  ({sum(score[1] * score[2] for score in scores)}/{sum(score[2] for score in scores)})")

    return per_config_scores, per_phenomenon_scores

