import torch
import locale

from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

locale.getpreferredencoding = lambda: "UTF-8"

# Enable mixed precision
scaler = torch.cuda.amp.GradScaler()



class ChatDoctor:
    def __init__(self, bnb_config, device_map):
        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            quantization_config = bnb_config,
            device_map = device_map
        )
        self.model.config.pretraining_tp = 1
        print("Loaded model...")

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Loaded tokenizer...")

    def inference(self, prompt, max_length: int = 500):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to('cuda')

        # Using greedy decoding and reducing the maximum length
        output = self.model.generate(input_ids, max_length=max_length)
        saying = self.tokenizer.decode(output[0])
        
        return saying