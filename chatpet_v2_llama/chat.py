import os
import sys
import torch
import fire
import time
import json
import argparse

from pathlib import Path
from typing import Tuple
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import *

def set_args():
    """
    Set arguments.
    1. device: GPU Setting
    2. temperature: Temperature for generation, default = 0
    3. model_path: pre_trained model path
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, required=False,
                        help='GPU Setting')
    parser.add_argument('--temperature', default=0, type=float, required=False,
                        help='Temperature for generation')
    parser.add_argument('--model_path', default='models/model_test', type=str, required=False,
                        help='pre_trained model path')
    return parser.parse_args()



def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()

    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = LLaMA_Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = LLaMA_Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def loulou_load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LouLou:
    start_time = time.time()

    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = LLaMA_Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = LLaMA_Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    chatbot = LouLou(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return chatbot


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    # Loading
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    # generator = load(
    #     ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    # )
    chatbot_loulou = loulou_load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    # Chatting
    while True:
        try:
            usr_in = input("User: ")
            chatbot_loulou.chat(
                usr_input = usr_in + '\n',
            )
        except KeyboardInterrupt:
            with open('./chat_history.txt', 'a') as f:
                f.write('\n-------------\n')
                f.write(chatbot_loulou.history)
            print('Saving Chat History')

if __name__ == "__main__":
    # Using fire.Fire() allows us to pass params from the command line
    fire.Fire(main)