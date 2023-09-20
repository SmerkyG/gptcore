import torch
import torch.utils.data
import lightning.pytorch.loggers
import optimizer
from dataset import MMapDataset
from util.config import Factory
import embed
import callback
import lightning.pytorch.callbacks
from lightning.pytorch.strategies import DeepSpeedStrategy
import lightning.pytorch.strategies
import datasets
import transformers
import dataset
import model
import model.hparams
import mask
import hparams
import cli
import dataset.tokenizer

import model.retnet

cli.Config(
    seed_everything = 1337,
    compile = True,
    model_factory = lambda: model.core.Decoder(
        hparams = model.hparams.HParams(
            n_layer=12,
            n_head=12,
            d_model=768,
            block_size=1024,

            feedforward_d_model_ratio=2,

            d_v_ratio=2,

            self_attention_sublayer_factory = lambda: model.hyena.HyenaAttentionSubLayer(),
            feedforward_sublayer_factory = lambda: model.core.RWKVChannelMix(),
        ),
    ),
    tokenizer_factory = lambda: transformers.AutoTokenizer.from_pretrained('gpt2'),
    dataset_transform_factories = [
        lambda: dataset.tokenizer.TokenizeMergeAndSplit()
    ],
    train_dataset_seed=32,
    train_dataset_factory = lambda: datasets.load_dataset(path='dataset/pile.py', streaming=True, split='train', subsets=['all']),
    val_dataset_factory = lambda: datasets.load_dataset(path='dataset/pile.py', streaming=True, split='validation', subsets=['all']),
    train_dataloader_factory = lambda: torch.utils.data.DataLoader(
        prefetch_factor=4, persistent_workers=True,
        batch_size = 8,
        num_workers = 4,
        pin_memory = True,
    ),
    val_dataloader_factory = lambda: torch.utils.data.DataLoader(
        batch_size = 8,
        pin_memory = True,
    ),
    fit_factory = lambda: lightning.Trainer.fit(
        #ckpt_path='checkpoints/epoch=0-step=256.ckpt',
    ),
    trainer_factory = lambda: lightning.Trainer(
        enable_progress_bar=False,
        #enable_checkpointing=False,
        max_epochs=-1,
        #val_check_interval=1024, # new
        precision = 'bf16-mixed',
        accumulate_grad_batches=1,
        gradient_clip_val=0.5,
        log_every_n_steps=5,
        logger = [
            #lightning.pytorch.loggers.CSVLogger(save_dir="."),
            lightning.pytorch.loggers.WandbLogger(project='pico', name='GPT2 L12D768H12CM2Adam'),
        ],
        callbacks = [
            lightning.pytorch.callbacks.ModelCheckpoint(), # should save after every validation anyway #every_n_train_steps=128
        ],
    ),
    optimizer_factories = lambda:
        torch.optim.Adam(
            lr=6e-4,
            betas=(0.9,0.999),
        ),
)

