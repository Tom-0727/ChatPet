# ChatPet - LouLou
This is a ChatPet based on GPT with Fine-tuning. 

## Introduction
ChatPet is an AI-based chatbot that aims to provide emotional value to users. The project focuses on creating a chatbot that can operate on normal devices with limited computing resources.

## Work Status
- LLaMA Deployment 🔲
  - Chat ✔️
    - Streaming Out :white_check_mark:
    - Multi Style 
  - Partial Params Finetune (LoRA) 🟢
    - PEFT
  - Short-Long Term Memory 🟢
    - Short-Term Memory :white_check_mark:
    - Long-Term Memory 

- GPT2 Deployment ☑️
  - Full Params Finetune    ✔️
  - Chat    ✔️
    - Chinese :white_check_mark:
    - English :white_check_mark:
  - Short-Term Memory ✔️
  - Repeatition Penalty, Temperature, Top-k, Top-p, accumulate_grad    ✔️


## How to use
### chatpet_v2_llama
```bash
cd chatpet_v2_llama
bash chat.sh  # Config your llama weights path
```
