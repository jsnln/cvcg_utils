import os
from rich import print
import time
from tqdm import tqdm
from glob import glob
import shutil
import torch
from build_objects import build_objects

from config import TrainerConfig

class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config

        self.model, self.dataset, self.dataloader, self.optimized_params, self.optimizer, self.lr_scheduler, self.loss_module, self.grad_scaler = build_objects(config)

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

        resume_from_state = None

        if self.config.resume_from_latest:
            ckpt_dir = os.path.join(self.config.out_dir, 'checkpoints')
            existing_latest_ckpts = glob(os.path.join(ckpt_dir, '*_latest.ckpt'))
            if len(existing_latest_ckpts) > 0:
                assert len(existing_latest_ckpts) == 1, f"there should only be one latest checkpoint in {ckpt_dir}, but found {len(existing_latest_ckpts)}"
                resume_from_state = existing_latest_ckpts[0]
                print(f'\\[resume info] resume_from_latest: true    -- we found one latest ckpt, this will override resume_from_state (if any) in your config')
            else:
                resume_from_state = self.config.resume_from_state
                print(f'\\[resume info] resume_from_latest: true    -- but no latest ckpt found. will use config.resume_from_state (if any) instead')


        if resume_from_state is not None:    # resume from a state checkpoint (the file must contain "model", "optimizer", "lr_scheduler", "cur_train_step" and "param_update_step")
            print(f'\\[resume info] trying to resume state from {resume_from_state}')
            checkpoint = torch.load(resume_from_state, map_location="cpu", weights_only=True)
            status = self.model.load_state_dict(checkpoint['model'], strict=False)
            print(f"\\[resume info] model loaded with status: {status}")

            if not self.config.reset_state:  # if True, load training states as well
                print(f"\\[resume info] reset_state is false. loading training states (optimizer/scheduler/cur_step) as well")
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                self.cur_train_step = checkpoint["cur_train_step"]
                self.cur_param_update_step = checkpoint["cur_param_update_step"]
                print(f"\\[resume info] training state resumed at cur_train_step: {self.cur_train_step}, cur_param_update_step: {self.cur_param_update_step}")
            else:
                print(f"\\[resume info] reset_state is true. not loading optimizer and scheduler states")
        
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
            if self.cur_train_step >= self.max_steps:
                print(f'\\[LOG] stopping: reached maximum number of steps {self.max_steps}')
                break

            progress_bar = tqdm(total=len(self.dataloader))
            progress_bar.set_description(f"Epoch {cur_epoch}")

            for local_step, data_batch in enumerate(self.dataloader):
                if self.cur_train_step >= self.max_steps: break

                # run model and get loss
                data_batch = self.preprocess_batch(data_batch)
                model_output = self.model(data_batch)
                total_loss, loss_dict, loss_str = self.loss_module(model_output)

                # check whether to update gradients
                update_grads = (self.cur_train_step + 1) % self.grad_accum_steps == 0 or self.cur_train_step == self.max_steps
                self.grad_scaler.scale(total_loss / self.grad_accum_steps).backward()

                skip_optimizer_step = False
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"NaN or Inf loss detected, skip this iteration")
                    skip_optimizer_step = True
                    total_loss = torch.zeros_like(total_loss)

                total_grad_norm = None
                # Check gradient norm and update optimizer if everything is fine
                if update_grads and (not skip_optimizer_step):
                    # Unscales the gradients
                    self.grad_scaler.unscale_(self.optimizer) 
                    # For all gradients, we safely change the NaN -> 0., inf -> 1e-6, -inf -> 1e-6.
                    with torch.no_grad():
                        for p in self.optimized_params:
                            if p.requires_grad and (p.grad is not None):
                                p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)
                
                    total_grad_norm = 0.0
                    if self.config.grad_clip_norm > 0:
                        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.optimized_params, max_norm=self.config.grad_clip_norm).item()
                        if total_grad_norm > self.config.skip_grad_threshold:
                            skip_optimizer_step = True
                            print(f"WARNING: step {self.cur_train_step} grad norm too large {total_grad_norm} > {self.config.skip_grad_threshold}, skipping optimizer step")

                if not skip_optimizer_step:
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    self.cur_param_update_step += 1
                self.cur_train_step += 1

                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                progress_bar.update(1)
                lr_str = f'lr: {self.lr_scheduler.get_last_lr()[0]:.3e}'
                log_str = f'[T/U]: [{self.cur_train_step, self.cur_param_update_step}] ' + loss_str + ', ' + lr_str
                progress_bar.set_postfix_str(log_str)
                
                if (self.cur_train_step % self.config.save_latest_every == 0):
                    # save latest model
                    save_path = os.path.join(self.config.out_dir, 'checkpoints', f'train_state_dict_{self.cur_train_step:06d}.ckpt')
                    self.save_train_state(save_path, mark_latest=True)
                    print(f'\\[checkpointing] saving train state to {save_path}')

                if (self.cur_train_step in self.config.extra_save_ckpt_at) \
                    or (self.cur_train_step % self.config.save_ckpt_every == 0):
                    
                    # save model
                    save_path = os.path.join(self.config.out_dir, 'checkpoints', f'train_state_dict_{self.cur_train_step:06d}.ckpt')
                    self.save_train_state(save_path, mark_latest=False)
                    print(f'\\[checkpointing] saving train state to {save_path}')

    def save_train_state(self, ckpt_path, mark_latest=False):
        """
        mark_latest: search for existing latest ckeckpoints saved in the same directory as `ckpt_path` and replace it with current checkpoint, also marked as latest
        """
        os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
        train_state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'cur_train_step': self.cur_train_step,
            'cur_param_update_step': self.cur_param_update_step,
        }

        if mark_latest:
            path_and_basename_noext, ext = os.path.splitext(ckpt_path)
            ckpt_path = path_and_basename_noext + '_latest' + ext
            existing_latest_ckpts = glob(os.path.join(os.path.dirname(ckpt_path), '*_latest' + ext))
            for existing_latest_path in existing_latest_ckpts:
                step_id = os.path.basename(existing_latest_path).split('_')[-2]
                step_id = int(step_id)
                if step_id <= self.cur_train_step:
                    os.remove(existing_latest_path)

        torch.save(train_state_dict, ckpt_path)



if __name__ == '__main__':
    config = TrainerConfig()
    trainer = Trainer(config)
    trainer.train()