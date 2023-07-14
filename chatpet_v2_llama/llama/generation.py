# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from llama.tokenizer import LLaMA_Tokenizer
from llama.model import LLaMA_Transformer


class LLaMA:
    def __init__(self, model: LLaMA_Transformer, tokenizer: LLaMA_Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        
        # Count the number of prompts, and make sure it's not exceed max batch size
        bsz = len(prompts)  # batch size
        params = self.model.params
        assert bsz <= params.max_batch_size, f'the prompts exceeds the max number for you prompt {bsz} but max is {params.max_batch_size}'

        # Tokenization to each prompt using the LLaMA_Tokenizer
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        # max_gen_len + max_prompt_size as total length but without exceeding max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        # Fullfil the tensor with pad tokens with the dimension of (batch_size, total_len)
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        # Fill the tensor with the prompt tokens
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        
        # get the mask info, which is the position of the prompt tokens
        input_text_mask = tokens != self.tokenizer.pad_id


        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            
            # Get the logits: raw, unnormalized outputs of a model 
            # before they have been transformed into probabilities using softmax

            # here is batch processing, namely, all prompt would be process simultaneously
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            # the tokens are predicted row by row, so it start from min_prompt_size
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
