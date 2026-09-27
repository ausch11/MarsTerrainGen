"""Lossless (within a configured bound) CLIP prompt chunking for SD1.5."""
import math
from collections import Counter

import torch


def prompt_token_ids(tokenizer, text):
    # tokenize/convert avoids the tokenizer's model_max_length truncation path.
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))


def audit_caption_lengths(records, tokenizer, chunk_length, max_chunks):
    """Check raw captions before expensive caching or optimizer initialization."""
    capacity = chunk_length - 2
    histogram = Counter()
    overflow = []
    largest = 0
    for record in records:
        text = record.caption.read_text(encoding="utf-8").strip()
        count = len(prompt_token_ids(tokenizer, text))
        needed = max(1, math.ceil(count / capacity))
        histogram[needed] += 1
        largest = max(largest, count)
        if needed > max_chunks and len(overflow) < 5:
            overflow.append((str(record.caption), count, needed))
    if overflow:
        raise ValueError(
            f"{sum(value for key, value in histogram.items() if key > max_chunks)} captions "
            f"exceed max_prompt_chunks={max_chunks} ({capacity} content tokens/chunk). "
            f"Examples: {overflow}. Increase max_prompt_chunks or shorten the captions."
        )
    return {"content_tokens_per_chunk": capacity, "max_content_tokens": largest,
            "chunks_per_caption": {str(k): histogram[k] for k in sorted(histogram)}}


def chunk_prompt(tokenizer, text, chunk_length, max_chunks, source="prompt"):
    if chunk_length < 3 or max_chunks < 1:
        raise ValueError("chunk_length must be >= 3 and max_chunks must be >= 1")
    tokens = prompt_token_ids(tokenizer, text)
    content_length = chunk_length - 2
    required = max(1, math.ceil(len(tokens) / content_length))
    if required > max_chunks:
        raise ValueError(
            f"{source}: {len(tokens)} CLIP content tokens require {required} chunks "
            f"of {content_length} tokens; max_prompt_chunks={max_chunks}. "
            "Increase max_prompt_chunks or shorten this text; no tokens were silently discarded.")
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    if bos is None or eos is None:
        raise ValueError("CLIP tokenizer must define BOS and EOS token IDs")
    rows = []
    for offset in range(0, len(tokens) or 1, content_length):
        piece = tokens[offset:offset + content_length]
        rows.append([bos, *piece, eos] + [eos] * (content_length - len(piece)))
    return torch.tensor(rows, dtype=torch.long)


def pad_prompt_chunks(items, tokenizer, chunk_length, chunks=None):
    """Return [batch, chunks, CLIP_length], padding with empty CLIP chunks."""
    target = chunks or max(item.shape[0] for item in items)
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    empty = torch.tensor([bos] + [eos] * (chunk_length - 1), dtype=torch.long)
    result = []
    for item in items:
        if item.shape[1] != chunk_length or item.shape[0] > target:
            raise ValueError("Incompatible CLIP chunk shape")
        extras = empty.expand(target - item.shape[0], -1)
        result.append(torch.cat([item, extras], dim=0))
    return torch.stack(result)


def encode_chunk_ids(text_encoder, input_ids):
    if input_ids.ndim == 2:
        input_ids = input_ids.unsqueeze(1)
    if input_ids.ndim != 3:
        raise ValueError("Expected [batch, chunks, CLIP_length] input IDs")
    batch, chunks, length = input_ids.shape
    max_positions = text_encoder.config.max_position_embeddings
    if length > max_positions:
        raise ValueError(f"CLIP chunk length {length} exceeds {max_positions} positions")
    hidden = text_encoder(input_ids.reshape(batch * chunks, length), return_dict=False)[0]
    return hidden.reshape(batch, chunks * length, hidden.shape[-1])


def encode_prompt_embeds(pipeline, prompts, max_chunks, chunk_length,
                         negative_prompts=None):
    """Encode all positive/negative tokens with equal sequence length for CFG."""
    if negative_prompts is not None and len(prompts) != len(negative_prompts):
        raise ValueError("Positive and negative prompt counts differ")
    positive = [chunk_prompt(pipeline.tokenizer, text, chunk_length, max_chunks,
                             source=f"prompt[{i}]") for i, text in enumerate(prompts)]
    negative = ([chunk_prompt(pipeline.tokenizer, text, chunk_length, max_chunks,
                              source=f"negative_prompt[{i}]")
                 for i, text in enumerate(negative_prompts)] if negative_prompts is not None else [])
    target = max(item.shape[0] for item in positive + negative)
    device = next(pipeline.text_encoder.parameters()).device
    positive_ids = pad_prompt_chunks(positive, pipeline.tokenizer, chunk_length, target).to(device)
    prompt_embeds = encode_chunk_ids(pipeline.text_encoder, positive_ids)
    if not negative:
        return prompt_embeds, None
    negative_ids = pad_prompt_chunks(negative, pipeline.tokenizer, chunk_length, target).to(device)
    negative_embeds = encode_chunk_ids(pipeline.text_encoder, negative_ids)
    return prompt_embeds, negative_embeds
