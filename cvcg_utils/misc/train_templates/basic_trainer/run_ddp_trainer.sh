torchrun --nproc_per_node 2 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:62840 \
    ddp_trainer.py
