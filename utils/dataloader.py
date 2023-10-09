# List of available datasets to concatenate
"""
datasets_names = [
    "medalpaca/medical_meadow_mediqa",
    "medalpaca/medical_meadow_mmmlu",
    "medalpaca/medical_meadow_medical_flashcards",
    "medalpaca/medical_meadow_wikidoc_patient_information",
    "medalpaca/medical_meadow_wikidoc",
    "medalpaca/medical_meadow_pubmed_causal",
    "medalpaca/medical_meadow_medqa",
    "medalpaca/medical_meadow_health_advice",
    "medalpaca/medical_meadow_cord19",
]
"""

# ======================= main ======================= #
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets


def get_dataset(datalist):
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load Dataset & Process
    datasets = [load_dataset(name, split="train") for name in datalist]
    combined_dataset = concatenate_datasets(datasets)
    def preprocess_function(x):
        return {
            "input_ids": tokenizer(x["instruction"] + " " + x["input"], truncation=True, max_length=512)["input_ids"],
            "labels": tokenizer(x["output"], truncation=True, max_length=512)["input_ids"],
        }
    processed_dataset = combined_dataset.map(preprocess_function)

    return  processed_dataset