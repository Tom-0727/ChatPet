from typing import List, Tuple

import torch

from llama.tokenizer import LLaMA_Tokenizer
from llama.model import LLaMA_Transformer


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


# Base LLaMA Class
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


            # print(self.tokenizer.decode(next_token.tolist()[0]))

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
        print(decoded)
        return decoded


# Lou Lou - ChatBot based on LLaMA
class LouLou(LLaMA):
    def __init__(self, model: LLaMA_Transformer, tokenizer: LLaMA_Tokenizer):
        super().__init__(model, tokenizer)

        with open('/home/tom/Tom_Files/iart_ai_lab/ChatPet/chatpet_v2_llama/prompts/chat_prompt.txt', 'r') as f:
            self.history = f.read()
        
        self.dialog_block = self.tokenizer.encode(self.history,
                                                  bos = True,
                                                  eos = False)

    def chat(self, 
             usr_input: str,
             stream: bool = False,
             temperature: float = 0.8,
             top_p: float = 0.95) -> str:
        params = self.model.params

        # Preprocessing
        self.history += usr_input
        input_tokens = self.tokenizer.encode(usr_input, bos = False, eos = False)
        self.dialog_block += input_tokens
        
        self.stm_manager() # Short Term Memory Apply

        num_tokens = len(self.dialog_block)

        tokens = torch.full((1, params.max_seq_len), self.tokenizer.pad_id).cuda().long()
        # Fill the tensor with the prompt tokens
        tokens[0, :len(self.dialog_block)] = torch.tensor(self.dialog_block).long()

        start_pos = num_tokens
        prev_pos = 0
        decode_block = []
        for cur_pos in range(start_pos, params.max_seq_len):
            
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            next_token = next_token.tolist()[0]
            
            decode_block.append(next_token)
            
            # BreakPoint, when an \n happens: \n == 13
            if next_token == 13:
                self.dialog_block.append(next_token)
                break
            
            # Streaming
            if stream:
                decoded = self.tokenizer.decode(next_token)
                print(decoded, end = '')

            # Update
            tokens[0, cur_pos] = next_token
            prev_pos = cur_pos
            self.dialog_block.append(next_token)
        
        self.history += self.tokenizer.decode(decode_block)
        if not stream:
            print(self.tokenizer.decode(decode_block))
        print('\n')
            
    # Short Term Memory Mechanism
    def stm_manager(self, 
                    back_step: int = 100,
                    margin: int = 50) -> None:
        max_len = self.model.params.max_seq_len
        if len(self.dialog_block) >= max_len-margin:
            self.dialog_block = self.dialog_block[back_step:]

