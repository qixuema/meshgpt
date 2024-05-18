from pathlib import Path
from functools import partial
from packaging import version
from contextlib import nullcontext, contextmanager

import numpy as np

import torch
# from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler

from pytorch_custom_utils import (
    get_adam_optimizer,
    # OptimizerWithWarmupSchedule,
    add_wandb_tracker_contextmanager
)

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, Tuple, Type, List

from ema_pytorch import EMA

from meshgpt_pytorch.data import custom_collate

from meshgpt_pytorch.version import __version__

from meshgpt_pytorch.meshgpt_pytorch import MeshAutoencoder

from meshgpt_pytorch.optimizer_scheduler import OptimizerWithScheduler

# constants

DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(
    find_unused_parameters = True
)

# helper functions

from meshgpt_pytorch.misc import (
    exists,
    # default,
    cycle,
    maybe_del,
    get_lr,
    # first,
    divisible_by,
)

# autoencoder trainer

@add_wandb_tracker_contextmanager()
class MeshAutoencoderTrainer(Module):
    @beartype
    def __init__(
        self,
        model: MeshAutoencoder,
        dataset: Dataset,
        num_train_steps: int = 10,
        batch_size: int = 1,
        num_workers: int = 8,
        grad_accum_every: int = 1,
        val_dataset: Optional[Dataset] = None,
        val_every_step: int = 1,
        val_num_batches: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        max_grad_norm: Optional[float] = 10,
        ema_kwargs: dict = dict(),
        scheduler: Optional[Type[LRScheduler]] = None,
        scheduler_kwargs: dict = dict(),
        accelerator_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        checkpoint_every_step = 1000,
        checkpoint_folder = './checkpoints_ae',
        data_kwargs: Tuple[str, ...] = ['vertices', 'faces', 'face_edges'],
        # warmup_steps = 1000,
        log_every_step = 10,
        use_wandb_tracking = False
    ):
        super().__init__()

        # experiment tracker

        self.use_wandb_tracking = use_wandb_tracking

        if use_wandb_tracking:
            accelerator_kwargs['log_with'] = 'wandb'
            
        self.log_every_step = log_every_step

        if 'kwargs_handlers' not in accelerator_kwargs:
            accelerator_kwargs['kwargs_handlers'] = [DEFAULT_DDP_KWARGS]

        # accelerator

        self.accelerator = Accelerator(**accelerator_kwargs)
        self.accelerator.print(f'device {str(self.accelerator.device)} is used!')

        self.model = model
        
        if self.is_main:
            # print(model)
            pass

        self.optimizer = OptimizerWithScheduler( # 初始化 base warmup 的时候，要注释掉 self.dampen
            accelerator = self.accelerator,
            optimizer = get_adam_optimizer(model.parameters(), lr = learning_rate, wd = weight_decay, **optimizer_kwargs),
            scheduler = scheduler,
            scheduler_kwargs = scheduler_kwargs,
            # warmup_steps = warmup_steps,
            max_grad_norm = max_grad_norm
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = num_workers,
            drop_last = True,
            collate_fn = partial(custom_collate, pad_id = model.pad_id)
        )

        self.should_validate = exists(val_dataset)

        self.val_every_step = val_every_step
        
        if self.should_validate and self.is_main:
            assert len(val_dataset) > 0, 'your validation dataset is empty'

            self.val_every_step = val_every_step
            self.val_num_batches = val_num_batches

            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size = batch_size,
                shuffle = True,
                # num_workers = num_workers,     
                num_workers = 0,
                drop_last = True,
                collate_fn = partial(custom_collate, pad_id = model.pad_id)
            )

        if hasattr(dataset, 'data_kwargs') and exists(dataset.data_kwargs):
            assert is_bearable(dataset.data_kwargs, List[str])
            self.data_kwargs = dataset.data_kwargs
        else:
            self.data_kwargs = data_kwargs

        (
            self.model,
            self.dataloader,
        ) = self.accelerator.prepare(
            self.model,
            self.dataloader,
        )

        if self.is_main:
            self.ema_model = EMA(model, **ema_kwargs)
            self.ema_model.to(self.accelerator.device) # 要在 accelerator.prepare 操作之后执行

        self.grad_accum_every = grad_accum_every
        self.num_train_steps = num_train_steps
        self.register_buffer('step', torch.tensor(0))
        # self.register_buffer('epoch', torch.tensor(0))

        self.checkpoint_every_step = checkpoint_every_step
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)

    @property
    def ema_tokenizer(self):
        return self.ema_model.ema_model

    def tokenize(self, *args, **kwargs):
        return self.ema_tokenizer.tokenize(*args, **kwargs)

    def log(self, **data_kwargs):
        self.accelerator.log(data_kwargs, step = self.step.item())

    @property
    def device(self):
        # return self.unwrapped_model.device
        return self.accelerator.device

    @property
    def local_process_index(self):
        return self.accelerator.local_process_index

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def print(self, msg):
        return self.accelerator.print(msg)

    def save(self, path, overwrite = True):
        path = Path(path)
        assert overwrite or not path.exists()

        pkg = dict(
            model = self.unwrapped_model.state_dict(),
            # model = self.model.state_dict(), 
            ema_model = self.ema_model.state_dict(),
            optimizer = self.optimizer.state_dict(),
            version = __version__,
            step = self.step.item(),
            # epoch = self.epoch.item(),
            config = self.unwrapped_model._config
        )

        torch.save(pkg, str(path))
        # self.accelerator.save_model(self.model, str(path))


    def load(self, path):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path))

        if version.parse(__version__) != version.parse(pkg['version']):
            self.print(f'loading saved line autoencoder at version {pkg["version"]}, but current package version is {__version__}')

        self.unwrapped_model.load_state_dict(pkg['model']) 
        
        if self.is_main:
            self.ema_model.load_state_dict(pkg['ema_model'])
            self.ema_model.to(self.device)
        
        self.optimizer.load_state_dict(pkg['optimizer'])

        self.step.copy_(pkg['step'])
        # self.epoch.copy_(pkg['epoch'])

    def next_data_to_forward_kwargs(self, dl_iter) -> dict:
        data = next(dl_iter)


        if isinstance(data, tuple):
            forward_kwargs = dict(zip(self.data_kwargs, data))

        elif isinstance(data, dict):
            forward_kwargs = data

        maybe_del(forward_kwargs, 'texts', 'text_embeds')
        return forward_kwargs

    def forward(self):
        step = self.step.item()
        dl_iter = cycle(self.dataloader)

        if self.is_main and self.should_validate:
            val_dl_iter = cycle(self.val_dataloader)

        while step < self.num_train_steps:
            # print('step is ', step)
            # print('self.step is ', self.step.item())
            
            # shuffle dataloader
            if divisible_by(step, self.val_every_step):
                self.dataloader.dataset.shuffle()            
            
            for i in range(self.grad_accum_every):
                is_last = i == (self.grad_accum_every - 1)
                maybe_no_sync = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

                forward_kwargs = self.next_data_to_forward_kwargs(dl_iter)

                with self.accelerator.autocast(), maybe_no_sync():

                    total_loss, (recon_loss, commit_loss) = self.model(
                        **forward_kwargs,
                        return_loss_breakdown = True
                    )

                    self.accelerator.backward(total_loss / self.grad_accum_every)

            if divisible_by(step, self.log_every_step):
                cur_lr = get_lr(self.optimizer.optimizer)
                
                print(f'lr: {cur_lr:.6f} | recon loss: {recon_loss.item():.3f} | commit loss: {commit_loss.sum().item():.3f}')
                
                self.log(
                    total_loss = total_loss.item(),
                    commit_loss = commit_loss.sum().item(),
                    recon_loss = recon_loss.item(),
                    cur_lr = cur_lr
                )

            # self.optimizer.step(step=step)
            self.optimizer.step()
            self.optimizer.zero_grad()

            step += 1
            self.step.add_(1)

            self.wait()

            if self.is_main:
                self.ema_model.update()

            self.wait()

            if self.is_main and self.should_validate and divisible_by(step, self.val_every_step):

                total_val_recon_loss = 0.
                self.ema_model.eval()

                num_val_batches = self.val_num_batches * self.grad_accum_every

                for _ in range(num_val_batches):
                    with self.accelerator.autocast(), torch.no_grad():

                        # try:
                        #     data = next(val_dl_iter)
                        #     print("Data loaded successfully:", data)
                        # except Exception as e:
                        #     print("Error loading data:", e)
                        
                        forward_kwargs = self.next_data_to_forward_kwargs(val_dl_iter)
                        forward_kwargs = {k: v.to(self.device) for k, v in forward_kwargs.items()} # 因为 val_dataloader 里面的数据是在 cpu 上的，所以要转到 gpu 上

                        val_loss, (val_recon_loss, val_commit_loss) = self.ema_model(
                            **forward_kwargs,
                            return_loss_breakdown = True
                        )

                        total_val_recon_loss += (val_recon_loss / num_val_batches)

                self.print(f'valid recon loss: {total_val_recon_loss:.3f}')

                self.log(val_loss = total_val_recon_loss)

            self.wait()

            if self.is_main and (divisible_by(step, self.checkpoint_every_step) or step == self.num_train_steps - 1):
                checkpoint_num = step // self.checkpoint_every_step
                self.save(self.checkpoint_folder / f'mesh-autoencoder.ckpt.{checkpoint_num}.pt')
                print(f'checkpoint saved at {self.checkpoint_folder / f"mesh-autoencoder.ckpt.{checkpoint_num}.pt"}')
                
            self.wait()
        
        # Make sure that the wandb tracker finishes correctly
        self.accelerator.end_training()
        
        self.print('training complete')      

    def val(self,):
        with torch.no_grad():
            # dl_iter = cycle(self.dataloader)
            val_dl_iter = cycle(self.val_dataloader)
            step = 0

            recon_faces_list = []
            while step < 5:
                forward_kwargs = self.next_data_to_forward_kwargs(val_dl_iter)
                forward_kwargs = {k: v.to(self.device) for k, v in forward_kwargs.items()} # 因为 val_dataloader 里面的数据是在 cpu 上的，所以要转到 gpu 上

                recon_faces = self.ema_model(
                    **forward_kwargs,
                    # return_loss_breakdown = True
                    only_return_recon_faces = True
                )
                step += 1
                recon_faces_list.append(recon_faces)

        return recon_faces_list
    
    def line2code(self, num_tokenize_steps):
        with torch.no_grad():
            # dl_iter = cycle(self.dataloader)
            dl_iter = cycle(self.dataloader)
            step = 0

            recon_faces_list = []
            while step < num_tokenize_steps:
                forward_kwargs = self.next_data_to_forward_kwargs(dl_iter)
                # forward_kwargs = {k: v.to(self.device) for k, v in forward_kwargs.items()} # 因为 val_dataloader 里面的数据是在 cpu 上的，所以要转到 gpu 上

                codes = self.model.module.tokenize(
                    **forward_kwargs,
                )
                current_process_idx = self.local_process_index

                # pytorch tensor to numpy array
                codes = codes.detach().cpu().numpy()            
                tgt_file_path = f'./codes/{current_process_idx}/{step}_codes.npy'
                np.save(tgt_file_path, codes)
                
                step += 1
                if self.is_main:
                    self.print(f'codes saved at {tgt_file_path}')
                
        self.print('tokenize complete')      
        
        return recon_faces_list

    def code2line(self, num_recon_steps):
        with torch.no_grad():
            # dl_iter = cycle(self.dataloader)
            dl_iter = cycle(self.dataloader)
            step = 0

            recon_faces_list = []
            while step < num_recon_steps:
                forward_kwargs = self.next_data_to_forward_kwargs(dl_iter)
                # forward_kwargs = {k: v.to(self.device) for k, v in forward_kwargs.items()} # 因为 val_dataloader 里面的数据是在 cpu 上的，所以要转到 gpu 上

                recon_faces = self.model.module.decode_from_codes_to_faces(
                    **forward_kwargs,
                )
                step += 1
                recon_faces_list.append(recon_faces)

        return recon_faces_list
