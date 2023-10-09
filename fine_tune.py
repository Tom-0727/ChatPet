import pandas as pd
import torch #deep learning
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
import locale
from peft import LoraConfig
from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling


locale.getpreferredencoding = lambda: "UTF-8"

# Enable mixed precision
scaler = torch.cuda.amp.GradScaler()

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype="float16", #halves the size of the model
        bnb_4bit_use_double_quant=False,
    )
device_map = {"": 0}


#https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        quantization_config=bnb_config,
        device_map=device_map
    )

model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def llama_inference(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
    # Using greedy decoding and reducing the maximum length
    output = model.generate(input_ids, max_length=500)
    return tokenizer.decode(output[0])



# ==================== Load and combine datasets ======================= #
# List of datasets to concatenate
datasets_names = [
    "medalpaca/medical_meadow_mediqa",
    "medalpaca/medical_meadow_mmmlu"
]

datasets = [load_dataset(name, split="train") for name in datasets_names]
combined_dataset = concatenate_datasets(datasets)

def preprocess_function(examples):
    return {
        "input_ids": tokenizer(examples["instruction"] + " " + examples["input"], truncation=True, max_length=512)["input_ids"],
        "labels": tokenizer(examples["output"], truncation=True, max_length=512)["input_ids"],
    }
processed_dataset = combined_dataset.map(preprocess_function)


# ==================== Training Config ======================= #
training_arguments = TrainingArguments(
    output_dir='results/',
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim='paged_adamw_32bit',
    save_steps=100,
    logging_steps=100,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=5000,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type='constant',
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

# Training
trainer = SFTTrainer(
    model=model,
    train_dataset=processed_dataset,
    peft_config=peft_config,
    dataset_text_field="input",
    max_seq_length=512,
    args=training_arguments,
    data_collator=data_collator,
    packing=False,
)
trainer.train(resume_from_checkpoint=True)

# Save model
model_save_path = './my_model'
trainer.save_model(model_save_path)

input_ids = tokenizer.encode('what is an allergy?', return_tensors="pt").to('cuda')
output = model.generate(input_ids, max_length=50)
generated_text = tokenizer.decode(output[0])

print(generated_text)