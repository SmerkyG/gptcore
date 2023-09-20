import torch
import torch.utils.data
import lightning.pytorch.loggers
import picogpt
import optimizer
from dataset import MMapDataset
from util.config import Factory, defer
import embed
import callback
import lightning.pytorch.callbacks
from lightning.pytorch.strategies import DeepSpeedStrategy
import lightning.pytorch.strategies
import datasets
import transformers
import lion
import dataset
import third_party
import mask
import hparams

from deepspeed.runtime.fp16.onebit.adam import OnebitAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.adam import FusedAdam

dict(
    seed_everything = 1337,
    #compile = True,
    model = core.Decoder(
        hparams = hparams.HParams(
            n_layer=12,
            n_head=12,
            d_model=768,#768,
            # n_layer=32,#30,
            # n_layer=4,
            # n_head=14,#16,#18,#12*3,#24,
            # d_model=28*128,#32*128,#18*128,#768*3,#2*768,
            block_size=1024,
            
            d_v_ratio=2,

            self_attention_sublayer_factory = 
                core.AttentionSubLayer(attention_factory = core.TorchAttention(mask_factory=mask.AlibiMask()),),
                #third_party.rwkv.RWKV5_TimeMix(),
                #third_party.hyena.HyenaSubLayer(),
            # self_attention_sublayer_factory = Factory(
            #     core.AttentionSubLayer,
            #     core.AttentionSubLayerConfig(
            #         d_v_ratio=2,
            #         attention_factory = Factory(core.TorchAttention, mask_factory=Factory(core.AlibiMask)),
            #     )
            # )
            rotary_positional_embedding_factory=Factory(embed.RotaryEmbedding),
            # attention_factory = Factory(core.TorchAttention, mask_factory=Factory(core.AlibiMask)),
            # d_v_ratio=2,
            feedforward_d_model_ratio=2,
            feedforward_sublayer_factory = core.RWKVChannelMix(),#Factory(core.SimplifiedChannelMix, activation_factory=Factory(torch.nn.SiLU)),
            #attention_sublayer_time_mixer_factory = Factory(core.TimeMixer),
            #final_norm_factory = Factory(core.RMSNorm), 
            #share_embedding_weights = False,
        )
    ),
    train_dataset = dataset.RandomDataset(),# datasets.load_dataset(path='pile.py', streaming=True, split='train'),
    val_dataset = dataset.RandomDataset(),#datasets.load_dataset(path='pile.py', streaming=True, split='validation'),
    #train_dataset = datasets.load_dataset(path='Skylion007/openwebtext', streaming=True, split='train'),
    # train_dataset = MMapDataset(
    #     data_dir = 'data/openwebtext',
    #     data_filename = 'train.bin',
    # ),
    train_dataloader = torch.utils.data.DataLoader(
        prefetch_factor=4, persistent_workers=True,
        batch_size = 8,
        num_workers = 4,
        pin_memory = True,
        #shuffle = True,
    ),
    val_dataloader = torch.utils.data.DataLoader(
        batch_size = 8,
        pin_memory = True,
    ),
    fit = lightning.Trainer.fit(
        #ckpt_path='checkpoints/epoch=0-step=256.ckpt',
    ),
    trainer = lightning.Trainer(
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
            #lightning.pytorch.loggers.WandbLogger(project='pico', name='L12D768H12CM2Adam'),
        ],
        callbacks = [
            lightning.pytorch.callbacks.ModelCheckpoint(), # should save after every validation anyway #every_n_train_steps=128
            #callback.PipelineSetupCallback(),
            #callback.GradAccumScheduleCallback(min=1, max=512//8, step=1, period=16_384_000, offset=8_192_000), #period=48_000_000, offset=24_000_000),#
            #lightning.pytorch.callbacks.RichProgressBar(),
        ],
        #deterministic=True,
        #strategy = DeepSpeedStrategy(stage=2, offload_optimizer=True, offload_parameters=True), #allgather_bucket_size=int(1e6), reduce_bucket_size=int(1e6), 
        #     config = {
        #         "zero_allow_untested_optimizer": True,
        #         # "optimizer": {
        #         #     "type": "Adam",
        #         #     "params": {
        #         #         "lr": 6e-4,
        #         #         "betas": [0.9, 0.999],
        #         #         "eps": 1e-8,
        #         #         "weight_decay": 0,
        #         #     },
        #         # },
        #         "zero_optimization": {
        #             "stage": 2,  # Enable Stage 2 ZeRO (Optimizer/Gradient state partitioning)
        #             "cpu_offload": True,  # Enable Offloading optimizer state/calculation to the host CPU
        #             "contiguous_gradients": True,  # Reduce gradient fragmentation.
        #             "overlap_comm": True,  # Overlap reduce/backward operation of gradients for speed.
        #             "allgather_bucket_size": 1e6,  # Number of elements to all gather at once.
        #             "reduce_bucket_size": 1e6,  # Number of elements we reduce/allreduce at once.
        #         },
        #     }            
        # )
        #devices=1,
        #strategy="ddp",
        #strategy='deepspeed_stage_2',
        #strategy = lightning.pytorch.strategies.FSDPStrategy(),
    ),
    #optimizer = transformers.Adafactor(lr=6e-4, relative_step=False),
    #optimizer = FusedAdam(lr=6e-4, betas=(0.9,0.999), bias_correction=True, adam_w_mode=False, amsgrad=False)
    #optimizer = OnebitAdam(lr=6e-4, betas=(0.9,0.999)),
    #optimizer = DeepSpeedCPUAdam(lr=6e-4, betas=(0.9,0.999), bias_correction=True, amsgrad=False),
    optimizer = 
        #lion.Lion()
        torch.optim.Adam(
        #optimizer.CompactAdamW(
            lr=6e-4,
            betas=(0.9,0.999),#(0.965, 0.99),#
            #weight_decay=0,#1e-2,
            #amsgrad=True,
            #fused=True,
        ),
)

