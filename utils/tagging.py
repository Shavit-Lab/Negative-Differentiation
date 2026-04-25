from typing import List, Tuple
import spacy

nlp = spacy.load("en_core_web_sm")

def align_pos_with_pretokenized(
    text_slice: str,
    window_input_ids: List[int],
    window_offsets: List[Tuple[int, int]],
    window_special_mask: List[int],
    nlp,
) -> Tuple[List[int], List[str], List[str]]:
    """
    Align POS/DEP for *pre-tokenized* HF tokens within a text slice.
    - text_slice: substring of the original full_text
    - window_input_ids: HF token ids for this window (already sliced)
    - window_offsets: offsets for each token, but **relative to text_slice** (0-based)
    - window_special_mask: 1 for special tokens, 0 otherwise
    """
    doc = nlp(text_slice)
    sp_spans = [(t.idx, t.idx + len(t), t.pos_, t.dep_) for t in doc]

    pos_tags, deps = [], []

    for (start, end), is_special in zip(window_offsets, window_special_mask):
        if is_special or end <= start:
            pos_tags.append("SPECIAL")
            deps.append("SPECIAL")
            continue

        best_pos, best_dep, best_overlap = "X", "", 0
        # Since spacy tokens are ordered, a simple scan is fine for clarity.
        # (Two-pointer optimization is possible if needed.)
        for s0, s1, p, d in sp_spans:
            if s0 >= end:
                break
            overlap = max(0, min(end, s1) - max(start, s0))
            if overlap > best_overlap:
                best_overlap = overlap
                best_pos, best_dep = p, d

        if best_overlap == 0:
            # Fallback: pick the spaCy token that contains `start` (rare)
            for s0, s1, p, d in sp_spans:
                if s0 <= start < s1:
                    best_pos, best_dep = p, d
                    best_overlap = s1 - s0
                    break

        pos_tags.append(best_pos if best_overlap > 0 else "X")
        deps.append(best_dep if best_overlap > 0 else "")

    return window_input_ids, pos_tags, deps

# %%
def iter_hf_windows_with_alignment(
    full_text: str,
    tokenizer,
    nlp,
    sequence_length: int = 2048,
):
    """
    Yields (input_ids, pos_tags, deps) per HF window of `sequence_length`.
    Uses a single full-text tokenization with offsets; for each window,
    parses only the minimal covering text slice in `full_text`.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Requires a fast tokenizer to get offset mappings.")

    enc = tokenizer(
        full_text,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        add_special_tokens=True,
        return_tensors=None,
    )

    all_ids = enc["input_ids"]
    all_offsets = enc["offset_mapping"]
    all_special = enc["special_tokens_mask"]

    if isinstance(all_ids[0], list):
        all_ids = all_ids[0]
        all_offsets = all_offsets[0]
        all_special = all_special[0]

    total_tokens = len(all_ids)
    num_full_windows = total_tokens // sequence_length

    for i in range(num_full_windows):
        start_tok = i * sequence_length
        end_tok = start_tok + sequence_length
        if end_tok > total_tokens:
            break

        window_input_ids = all_ids[start_tok:end_tok]
        window_offsets_abs = all_offsets[start_tok:end_tok]
        window_special_mask = all_special[start_tok:end_tok]

        # Compute minimal covering char span over NON-SPECIAL tokens that have real spans
        # (specials often have (0,0) or nonsensical offsets).
        non_special_spans = [(s, e) for (s, e), m in zip(window_offsets_abs, window_special_mask)
                             if m == 0 and e > s]
        if non_special_spans:
            char_start = min(s for s, _ in non_special_spans)
            char_end   = max(e for _, e in non_special_spans)
        else:
            char_start, char_end = 0, 0

        text_slice = full_text[char_start:char_end]

        # Relativize offsets to the slice start, keep specials as-is (usually (0,0))
        window_offsets_rel = []
        for (s, e) in window_offsets_abs:
            if e > s:
                window_offsets_rel.append((max(0, s - char_start), max(0, e - char_start)))
            else:
                window_offsets_rel.append((0, 0))

        # Align using pretokenized data
        ids, pos_tags, deps = align_pos_with_pretokenized(
            text_slice=text_slice,
            window_input_ids=window_input_ids,
            window_offsets=window_offsets_rel,
            window_special_mask=window_special_mask,
            nlp=nlp,
        )

        yield ids, pos_tags, deps