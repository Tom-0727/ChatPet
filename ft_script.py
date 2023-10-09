import os
import torch
import locale
import argparse
import pandas as pd

from trl import SFTTrainer
from peft import LoraConfig
from utils import get_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling


locale.getpreferredencoding = lambda: "UTF-8"

# Enable mixed precision
scaler = torch.cuda.amp.GradScaler()



def set_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_ckpt_path', type=str, default='./logger/ckpt/undefined', required=False,
                        help='path to save checkpoint')
    parser.add_argument('--save_model_path', type=str, default='./model/undefined', required=False,
                        help='path to save model')
    parser.add_argument('--save_steps', type=int, default=100, required=False,
                        help='save model every x steps')
    parser.add_argument('--continue_train', action='store_true',
                        help='continue training from checkpoint')
    
    args = parser.parse_args()
    return args


def fine_tune(args):
    # ==================== Get Dataset ======================= #
    print('Now loading dataset...')
    datalist = [
        "medalpaca/medical_meadow_mediqa",
        "medalpaca/medical_meadow_mmmlu"
    ]
    dataset = get_dataset(datalist)


    # ==================== Load Model ======================= #
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype="float16", #halves the size of the model
            bnb_4bit_use_double_quant=False,
        )
    device_map = {"": 0}

    print('Now loading model...')
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        quantization_config = bnb_config,
        device_map = device_map
    )


    # ==================== Load Tokenizer ======================= #
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token


    # ==================== Fine Tuning Config ======================= #
    training_arguments = TrainingArguments(
        output_dir = args.save_ckpt_path,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        optim = 'paged_adamw_32bit',
        save_steps = 100,
        logging_steps = 100,
        learning_rate = 2e-4,
        fp16 = True,
        max_grad_norm = 0.3,
        max_steps = 5000,
        warmup_ratio = 0.03,
        group_by_length = True,
        lr_scheduler_type = 'constant',
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Define data collator to handle tokenization and collation
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )


    # ==================== Start Fine Tuning ======================= #
    print('Now start fine tuning...')
    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        peft_config = peft_config,
        dataset_text_field = "input",
        max_seq_length = 512,
        args = training_arguments,
        data_collator = data_collator,
        packing = False,
    )

    if args.continue_train:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # ==================== Save Model ======================= #
    print('Now saving model...')
    trainer.save_model(args.save_model_path)
    print('Model saved.')


def main():
    # Set Arguments
    args = set_args()
    os.makedirs(args.save_ckpt_path, exist_ok=True)
    os.makedirs(args.save_model_path, exist_ok=True)

    # Broadcast
    if args.continue_train:
        print("Continue training from checkpoint...")
    else:
        print("Start training from scratch...")

    # Start Fine Tuning
    fine_tune(args)


if __name__ == '__main__':
    main()
