# Trainer Templates

## Progress

- [ ] Basic trainer
    - [x] Modulization
        - [x] model
        - [x] optimized_params, optimizer
        - [x] loss_module
        - [x] grad_scaler
        - [x] lr_scheduler
        - [x] dataset and dataloader
        - [ ] logger
    - [x] Training tricks
        - [ ] AMP
        - [x] warmup
        - [x] grad norm clip
        - [x] grad update skip
    - [x] Auto resume
    - [ ] Evaluate
        - [ ] model evaluate
        - [ ] wandb logging
        - [x] model saving

## Tutorial

1. The base classes are only used for type hints. We do not inherite them