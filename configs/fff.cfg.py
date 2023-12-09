import torch
import torch.utils.data
import lightning.pytorch.loggers
import lightning.pytorch.callbacks
import lightning.pytorch.strategies
import transformers
import dataset
import model.hparams
import cli
import dataset.tokenizer
import lit

import model.core
import model.llama
import model.fff
import posemb

BATCH_SIZE = 40
VOCAB_SIZE = 32000
TOKENIZER_FACTORY = lambda: transformers.AutoTokenizer.from_pretrained(
    'mistralai/Mistral-7B-v0.1'
)
MAX_SEQUENCE_LENGTH = 1024

LOG_PROJECT = 'gptcore'
LOG_NAME = 'FFF B40L12D768Q16V4 AdamW'

cli.Config(
    seed_everything = 1337,
    compile = False,

    model_factory = lambda: model.core.Decoder(
        hparams = model.hparams.HParams(
            vocab_size = VOCAB_SIZE,
            max_sequence_length=MAX_SEQUENCE_LENGTH,

            n_layer=12,
            n_head=16,
            n_leaf=32,
            d_model=768,

            feedforward_d_model_ratio=8/3,
            n_kv_head_ratio=0.25,

            d_v_ratio=1,

            rotary_positional_embedding_factory = lambda sequence_length, d_query: (
                posemb.RotaryEmbedding(sequence_length, d_query)
            ),
        ),
        layer_factory=lambda: model.core.GradientCheckpointing(
            module_factory = lambda: model.core.TransformerLayer(
                self_attention_sublayer_factory = lambda: model.llama.Llama2AttentionSubLayer(),
                feedforward_sublayer_factory = lambda: model.llama.Llama2FeedForwardSubLayer(),
            ),
        )
    ),

    trainer_factory = lambda: lit.CoreLightningTrainer(
        optimizer_factory = lambda params: torch.optim.AdamW(
            params=params,
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=0.1
        ),
        lightning_trainer_factory = lambda: lightning.Trainer(
            enable_progress_bar=False,
            #enable_checkpointing=False,
            max_epochs=-1,
            #val_check_interval=1024, # new
            precision = 'bf16-mixed',
            accumulate_grad_batches=1,
            gradient_clip_val=1.0,
            log_every_n_steps=5,
            logger = [
                #lightning.pytorch.loggers.CSVLogger(save_dir="."),
                lightning.pytorch.loggers.WandbLogger(project=LOG_PROJECT, name=LOG_NAME),
            ]
        ),
        datamodule_factory=lambda: dataset.DM(
            dataset_path='dataset/pile.py', 
            tokenizer_factory=TOKENIZER_FACTORY, 
            batch_size=BATCH_SIZE, 
            sequence_length=MAX_SEQUENCE_LENGTH, 
            num_workers=4,
            seed=32,
        ),
    ),
)

