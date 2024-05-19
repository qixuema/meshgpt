import gc
import datetime
from pathlib import Path
from functools import partial
from packaging import version
from contextlib import nullcontext, contextmanager

import torch
# from torch import nn, Tensor
from torch.nn import Module
# import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler

from einops import rearrange

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

# from ema_pytorch import EMA

from meshgpt_pytorch.data import custom_collate

from meshgpt_pytorch.version import __version__

from meshgpt_pytorch.meshgpt_pytorch import MeshTransformer

from meshgpt_pytorch.optimizer_scheduler import OptimizerWithScheduler

from meshgpt_pytorch.trainer_tools import save_checkpoint

# constants

DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(
    find_unused_parameters = True
)

from meshgpt_pytorch.misc import (
    exists,
    # default,
    cycle,
    get_lr,
    divisible_by,
)

from utils.utils import face_coords_to_file

from utils.utils import load_ae


# mesh transformer trainer

@add_wandb_tracker_contextmanager()
class MeshTransformerTrainer(Module):
    @beartype
    def __init__(
        self,
        model: MeshTransformer,
        dataset: Dataset,
        num_train_steps: int = 100,
        batch_size: int = 1,
        num_workers: int = 8,
        grad_accum_every: int = 1,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.,
        max_grad_norm: Optional[float] = 0.5,
        val_dataset: Optional[Dataset] = None,
        val_every_step = 1,
        val_num_batches = 5,
        scheduler: Optional[Type[LRScheduler]] = None,
        scheduler_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        accelerator_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        checkpoint_every_setp = 1000,
        checkpoint_folder = './checkpoints_transformer',
        data_kwargs: Tuple[str, ...] = ['vertices', 'faces', 'face_edges', 'room_codes', 'text_embeds'],
        # warmup_steps = 1000,
        log_every_step = 10,
        use_wandb_tracking = False,
        first_train_transformer: bool = False,
        ae_ckpt_path: str = '',
    ):
        super().__init__()

        # experiment tracker

        self.use_wandb_tracking = use_wandb_tracking

        if use_wandb_tracking:
            accelerator_kwargs['log_with'] = 'wandb'

        self.log_every_step = log_every_step


        if 'kwargs_handlers' not in accelerator_kwargs:
            accelerator_kwargs['kwargs_handlers'] = [DEFAULT_DDP_KWARGS]

        self.accelerator = Accelerator(**accelerator_kwargs)
        self.accelerator.print(f'device {str(self.accelerator.device)} is used!')

        if first_train_transformer:
            model.autoencoder = load_ae(model.autoencoder, ae_ckpt_path, self.accelerator.device)
            print("load autoencoder on device: ", self.accelerator.device)

        self.model = model

        optimizer = get_adam_optimizer(
            model.parameters(),
            lr = learning_rate,
            wd = weight_decay,
            filter_by_requires_grad = True, # filter ae model params
            **optimizer_kwargs
        )

        self.optimizer = OptimizerWithScheduler(
            accelerator = self.accelerator,
            optimizer = optimizer,
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

            self.val_num_batches = val_num_batches

            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size = batch_size,
                shuffle = True,
                num_workers = num_workers,                
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
        
        # if self.is_main:
        #     self.ema_model = EMA(model, **ema_kwargs)
        #     self.ema_model.to(self.device) # 要在 accelerator.prepare 操作之后执行        

        self.grad_accum_every = grad_accum_every
        self.num_train_steps = num_train_steps
        self.register_buffer('step', torch.tensor(0))

        self.checkpoint_every_step = checkpoint_every_setp
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)

    # @property
    # def ema_tokenizer(self):
    #     return self.ema_model.ema_model

    # def tokenize(self, *args, **kwargs):
    #     return self.ema_tokenizer.tokenize(*args, **kwargs)

    def log(self, **data_kwargs):
        self.accelerator.log(data_kwargs, step = self.step.item())

    @property
    def device(self):
        return self.unwrapped_model.device

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

    def next_data_to_forward_kwargs(self, dl_iter) -> dict:
        data = next(dl_iter)

        if isinstance(data, tuple):
            forward_kwargs = dict(zip(self.data_kwargs, data))

        elif isinstance(data, dict):
            forward_kwargs = data

        return forward_kwargs

    def save(self, path, overwrite = True):
        path = Path(path)
        assert overwrite or not path.exists()

        pkg = dict(
            model = self.unwrapped_model.state_dict(),
            # model = self.model.state_dict(),
            optimizer = self.optimizer.state_dict(),
            step = self.step.item(),
            version = __version__
        )

        # torch.save(pkg, str(path))
        save_checkpoint(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path), map_location=self.device)

        if version.parse(__version__) != version.parse(pkg['version']):
            self.print(f'loading saved mesh transformer at version {pkg["version"]}, but current package version is {__version__}')

        self.unwrapped_model.load_state_dict(pkg['model'])
        self.optimizer.load_state_dict(pkg['optimizer'])
        self.step.copy_(pkg['step'])
        del pkg
        gc.collect()
        

    def forward(self):
        step = self.step.item()
        dl_iter = cycle(self.dataloader)

        is_multi_loss = False

        if self.should_validate and self.is_main:
            val_dl_iter = cycle(self.val_dataloader)

        while step < self.num_train_steps:
            
            # shuffle dataloader
            if divisible_by(step, self.val_every_step):
                self.dataloader.dataset.shuffle()     
                if self.is_main:
                    self.val_dataloader.dataset.shuffle()       

            for i in range(self.grad_accum_every):
                is_last = i == (self.grad_accum_every - 1)
                maybe_no_sync = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

                forward_kwargs = self.next_data_to_forward_kwargs(dl_iter)

                with self.accelerator.autocast(), maybe_no_sync():
                    loss = self.model(**forward_kwargs)

                    if isinstance(loss, tuple):
                        is_multi_loss = True
                        loss_1, loss_2, loss_weight = loss
                        loss = loss_weight * loss_1 + (1 - loss_weight) * loss_2

                    self.accelerator.backward(loss / self.grad_accum_every)

            if divisible_by(step, self.log_every_step):
                cur_lr = get_lr(self.optimizer.optimizer)

                # self.print(f'step: {step}, lr: {cur_lr:.7f}, loss: {loss.item():.3f}')

                if is_multi_loss:
                    self.log(
                        loss = loss.item(),
                        loss_1 = loss_1.item(),
                        loss_2 = loss_2.item(),
                        cur_lr = cur_lr,)
                else:
                    self.log(
                        loss = loss.item(),
                        cur_lr = cur_lr,)


            self.optimizer.step()
            self.optimizer.zero_grad()

            step += 1
            self.step.add_(1)

            self.wait()

            if self.is_main and self.should_validate and divisible_by(step, self.val_every_step):

                total_val_loss = 0.
                self.unwrapped_model.eval()

                num_val_batches = self.val_num_batches * self.grad_accum_every

                for _ in range(num_val_batches):
                    with self.accelerator.autocast(), torch.no_grad():

                        forward_kwargs = self.next_data_to_forward_kwargs(val_dl_iter)
                        forward_kwargs = {k: v.to(self.device) for k, v in forward_kwargs.items()} # 因为 val_dataloader 里面的数据是在 cpu 上的，所以要转到 gpu 上
                        
                        val_loss = self.unwrapped_model(**forward_kwargs)

                        if isinstance(val_loss, tuple):
                            val_loss_1, val_loss_2, loss_weight = val_loss
                            val_loss = loss_weight * val_loss_1 + (1 - loss_weight) * val_loss_2

                        total_val_loss += (val_loss / num_val_batches)

                self.print(f'valid recon loss: {total_val_loss:.3f}')

                self.log(val_loss = total_val_loss)

            self.wait()

            if self.is_main and (divisible_by(step, self.checkpoint_every_step) or step == self.num_train_steps - 1):
                checkpoint_num = step // self.checkpoint_every_step
                idx = str(checkpoint_num).zfill(2)
                self.save(self.checkpoint_folder / f'mesh-transformer.ckpt.{idx}.pt')
                # 获取当前时间
                current_time = datetime.datetime.now()
                # 格式化输出时间
                formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                print(formatted_time, f' checkpoint saved at {self.checkpoint_folder / f"mesh-transformer.ckpt.{idx}.pt"}')

            self.wait()

        # Make sure that the wandb tracker finishes correctly
        self.accelerator.end_training()

        self.print('training complete')
    