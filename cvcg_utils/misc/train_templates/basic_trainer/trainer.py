import os
from rich import print
import time
from tqdm import tqdm

import torch
from build_objects import build_objects

from config import TrainerConfig

class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config

        self.model, self.dataset, self.dataloader, self.optimized_params, self.optimizer, self.lr_scheduler, self.loss_module = build_objects(config)

        # this is where things differ
        # please prepare all objects according to no-dist / ddp / accelerate
        self.model.cuda()
        self.loss_module.cuda()

        # training setup and state
        # hyperparameters
        self.max_steps = config.max_steps
        self.grad_accum_steps = config.grad_accum_steps
        self.max_steps = self.max_steps * self.grad_accum_steps # real train steps when using gradient accumulation
        self.total_batch_size = config.batch_size   # * world_size if using ddp

        # state
        self.cur_train_step = 0
        self.cur_param_update_step = 0
        
        def to_cuda(batch):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            return batch

        self.preprocess_batch = to_cuda

        # resume
        self.try_to_resume()
        

    def try_to_resume(self):
        if self.config.resume_from_state is not None:    # resume from a state checkpoint (the file must contain "model", "optimizer", "lr_scheduler", "cur_train_step" and "param_update_step")
            print(f'\\[resume info] trying to resume state from {self.config.resume_from_state}')
            checkpoint = torch.load(self.config.resume_from_state, map_location="cpu")
            status = self.model.load_state_dict(checkpoint['model'], strict=False)
            print(f"\\[resume info] model loaded with status: {status}")

            if not self.config.reset_state:  # if True, load training states as well
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                print(f"\\[resume info] optimizer loaded")
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                print(f"\\[resume info] lr scheduler loaded")
                self.cur_train_step = checkpoint["cur_train_step"]
                self.cur_param_update_step = checkpoint["param_update_step"]
                print(f"\\[resume info] training state resumed at cur_train_step: {self.cur_train_step}, cur_param_update_step: {self.cur_param_update_step}")
        
        elif self.config.resume_from_model is not None:  # resume from a checkpoint that contain only model weights
            print(f'\\[resume info] trying to resume from a pretrained model: {self.config.resume_from_model}')
            checkpoint = torch.load(self.config.resume_from_model, map_location="cpu")
            status = self.model.load_state_dict(checkpoint, strict=False)
            print(f"\\[resume info] model loaded with status: {status}")

        else:
            print(f"\\[resume info] starting from scratch")


    def train(self):
        

        self.model.activate_trainale_parameters()
        
        start_epoch = int(self.cur_train_step * (self.total_batch_size / self.grad_accum_steps) // len(self.dataset) )

        for cur_epoch in range(start_epoch, self.config.num_epochs):  # not exactly restarting from that iteration, a possible fix is to load sampler state

            progress_bar = tqdm(total=len(self.dataloader))
            progress_bar.set_description(f"Epoch {cur_epoch}")

            for step, data_batch in enumerate(self.dataloader):

                data_batch = self.preprocess_batch(data_batch)

                model_output = self.model(data_batch)
                total_loss, loss_dict, loss_str = self.loss_module(model_output)
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.optimized_params, self.config.grad_clip_norm)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                progress_bar.update(1)
                lr_str = f'lr: {self.lr_scheduler.get_last_lr()[0]:.3e}'
                log_str = f'[T/U]: [{self.cur_train_step, self.cur_param_update_step}] ' + loss_str + ', ' + lr_str
                progress_bar.set_postfix_str(log_str)
                
                # self.accelerator.log(logs, step=global_step) logging tool
                
                self.cur_train_step += 1
                self.cur_param_update_step += 1

                if self.cur_train_step == 1 or self.cur_train_step % 1000 == 0:
                    pass
                    # evaluate and save models

if __name__ == '__main__':
    config = TrainerConfig()
    trainer = Trainer(config)
    trainer.train()