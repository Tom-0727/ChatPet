# torchrun equal to python -m torch.distributed.launch
torchrun --nproc_per_node 1 chat.py \
    --ckpt_dir /home/tom/Tom_Files/iart_ai_lab/weights/7B \
    --tokenizer_path /home/tom/Tom_Files/iart_ai_lab/weights/tokenizer,model
