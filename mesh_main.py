import os
import sys
from torch.optim.lr_scheduler import LambdaLR

from meshgpt_pytorch import (
    MeshAutoencoder,
    Dataset,
)

from argparse import ArgumentParser
from utils.config import NestedDictToClass, load_config

os.environ['HTTP_PROXY'] = 'http://172.31.178.184:7788'
os.environ['HTTPS_PROXY'] = 'http://172.31.178.184:7788'

# Arguments
program_parser = ArgumentParser(description='Train a line autoencoder model.')
program_parser.add_argument('--config', type=str, default='', help='Path to config file.')
program_args = program_parser.parse_args()

cfg = load_config(program_args.config)
args = NestedDictToClass(cfg)

isDebug = True if sys.gettrace() else False

train_dataset = Dataset(
    is_single = args.data.is_single,
    dataset_file_path = args.data.train_set_file_path,
    # condition_on_text = args.condition_on_text,
    # add_room_codes = args.add_room_codes,
    replica=args.data.replication,
    is_train=True)
val_dataset = Dataset(
    is_single = args.data.is_single,
    dataset_file_path = args.data.test_set_file_path,
    # condition_on_text = args.condition_on_text,
    # add_room_codes = args.add_room_codes,
    replica=args.data.replication,
    is_train=False)

# autoencoder

autoencoder = MeshAutoencoder(
    # use_linear_attn = args.model.use_linear_attn,
    use_residual_lfq = args.model.use_residual_lfq,
    encoder_dims_through_depth = args.model.encoder_dims_through_depth,
    decoder_dims_through_depth = args.model.decoder_dims_through_depth,
    codebook_size = args.model.codebook_size,
    attn_encoder_depth = args.model.attn_encoder_depth,
    attn_decoder_depth = args.model.attn_decoder_depth,
)

train_ae = args.train_ae
if train_ae:
    from meshgpt_pytorch import MeshAutoencoderTrainer

    from utils.utils import load_ae
    if not args.first_train_ae:
        autoencoder = load_ae(autoencoder, args.model.ae_ckpt_path)
    
    epochs = args.epochs
    batch_size = args.batch_size
    num_gpus = args.num_gpus
    num_step_per_epoch = int(train_dataset.__len__() / (batch_size * num_gpus))
    num_train_steps = epochs * num_step_per_epoch
    num_warmup_steps = int(0.1*num_train_steps)
    
    autoencoder_trainer = MeshAutoencoderTrainer(
        autoencoder,
        dataset = train_dataset,
        val_dataset = val_dataset,
        num_train_steps = num_train_steps,
        batch_size = batch_size,
        num_workers = args.num_workers,
        grad_accum_every = 2,
        val_every_step = args.val_every_epoch * num_step_per_epoch,
        val_num_batches = 100,
        learning_rate = args.lr,
        max_grad_norm = 20.,
        scheduler = LambdaLR,
        scheduler_kwargs = dict(
            num_warmup_steps = num_warmup_steps,
            num_training_steps = num_train_steps,),
        checkpoint_every_step = args.save_every_epoch * num_step_per_epoch,
        accelerator_kwargs = dict(
            cpu = False,
            step_scheduler_with_optimizer=False
        ),
        log_every_step = 1,
        use_wandb_tracking = True,
        checkpoint_folder = args.model.ae_checkpoint_folder,
    )
    
    if autoencoder_trainer.is_main:
        total_params = sum(p.numel() for p in autoencoder.encoders.parameters())
        print(f"encoders Total parameters: {total_params / 1e6} M")
        total_params = sum(p.numel() for p in autoencoder.encoder_attn_blocks.parameters())
        print(f"encoder_attn Total parameters: {total_params / 1e6} M")
        total_params = sum(p.numel() for p in autoencoder.decoders.parameters())
        print(f"decoders Total parameters: {total_params / 1e6} M")  
        total_params = sum(p.numel() for p in autoencoder.decoder_attn_blocks.parameters())
        print(f"decoder_attn Total parameters: {total_params / 1e6} M")      
        total_params = sum(p.numel() for p in autoencoder.parameters())
        print(f"autoencoder Total parameters: {total_params / 1e6} M")  

    if not isDebug and autoencoder_trainer.is_main:
        print("not debug")
        # autoencoder_trainer.load('checkpoints/line-autoencoder.ckpt.7.pt')

        if autoencoder_trainer.use_wandb_tracking:
            # Initialise your wandb run, passing wandb parameters and any config information
            autoencoder_trainer.accelerator.init_trackers(
                project_name=args.wandb_project_name, 
                config=cfg,
                init_kwargs={"wandb": {"name": args.wandb_name}}
            )

    autoencoder_trainer()

    exit(0)

train_transformer = args.train_transformer

# from utils.utils import load_ae
# if args.first_train_transformer:
#     autoencoder = load_ae(autoencoder, args.model.ae_ckpt_path)

if train_transformer:
    from meshgpt_pytorch import (
        MeshTransformerTrainer,
        MeshTransformer,
    )
    
    transformer = MeshTransformer(
        autoencoder,
        dim = args.model.dim,
        attn_heads = args.model.attn_heads,
        max_seq_len = args.model.max_seq_len,
        coarse_pre_gateloop_depth = args.model.coarse_pre_gateloop_depth,
        fine_pre_gateloop_depth = args.model.fine_pre_gateloop_depth,
        condition_on_text = args.condition_on_text,
        use_text_embeds = args.use_text_embeds,
        add_room_codes = args.add_room_codes,
    )
    
    epochs = args.epochs
    batch_size = args.batch_size
    num_gpus = args.num_gpus
    num_step_per_epoch = int(train_dataset.__len__() / (batch_size * num_gpus))
    num_train_steps = epochs * num_step_per_epoch
    num_warmup_steps = int(0.1*num_train_steps)
        
    transformer_trainer = MeshTransformerTrainer(
        transformer,
        dataset = train_dataset,
        val_dataset = val_dataset,
        num_train_steps = num_train_steps,
        batch_size = batch_size,
        num_workers = args.num_workers,
        grad_accum_every = 2,
        learning_rate = args.lr,
        scheduler = LambdaLR,
        scheduler_kwargs = dict(
            num_warmup_steps = num_warmup_steps,
            num_training_steps = num_train_steps,),
        val_every_step = args.val_every_epoch * num_step_per_epoch,
        checkpoint_every_setp = args.save_every_epoch * num_step_per_epoch,
        log_every_step = args.log_every_step,
        accelerator_kwargs = dict(
            cpu = False,
            step_scheduler_with_optimizer=False
        ),
        use_wandb_tracking = args.use_wandb_tracking,
        checkpoint_folder = args.model.transformer_checkpoint_folder,
        first_train_transformer = args.first_train_transformer,
        ae_ckpt_path=args.model.ae_ckpt_path,
        
    )


    if not args.first_train_transformer:
        transformer_trainer.load(args.model.transformer_ckpt_path)
        print("load transformer")

    if transformer_trainer.is_main:
        total_params = sum(p.numel() for p in transformer.parameters())
        print(f"Total parameters: {total_params / 1e6}") 
        # 统计不参与梯度更新的参数量
        non_trainable_params = sum(p.numel() for p in transformer.parameters() if not p.requires_grad)
        print(f"Number of non-trainable parameters: {non_trainable_params/ 1e6}")

    if not isDebug and transformer_trainer.is_main:
        print("not debug")
        # transformer_trainer.load('checkpoints/line-autoencoder.ckpt.7.pt')

        if transformer_trainer.use_wandb_tracking:
            # Initialise your wandb run, passing wandb parameters and any config information
            transformer_trainer.accelerator.init_trackers(
                project_name=args.wandb_project_name,
                config=cfg,
                init_kwargs={"wandb": {"name": args.wandb_name}}
            )

    transformer_trainer()
    
    exit(0)