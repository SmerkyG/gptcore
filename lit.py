from typing import Any, Optional, Tuple

import time
import math

import lightning
import lightning as L
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import torch.optim.lr_scheduler
import torch.utils.data.dataset
import scheduler

import torch.backends.cuda
import torch.backends.cudnn

import metrics

import model.core

from lightning import LightningModule

from util.config import Factory

from dataclasses import dataclass, field

from typing import Callable, Any

import cli

def field_default(fn):
    return field(default_factory=fn)

def collate_target_tokens_offset_by_one_input_ids(batch): 
    tuple_batch = [(d['input_ids'][:-1], d['input_ids'][1:]) for d in batch]
    return torch.utils.data.default_collate(tuple_batch)

@dataclass
class CoreLightningTrainer(cli.ITrainer):
    train_dataset_factory:Callable[..., torch.utils.data.dataset.Dataset]=field_default(lambda: Factory())
    train_dataloader_factory:Callable[..., torch.utils.data.DataLoader]=field_default(lambda: Factory())
    val_dataset_factory:Callable[..., torch.utils.data.dataset.Dataset]=field_default(lambda: Factory())
    val_dataloader_factory:Callable[..., torch.utils.data.DataLoader]=field_default(lambda: Factory())
    datamodule_factory:Callable[..., lightning.LightningDataModule]|None=None
    optimizer_factory:Callable[..., torch.optim.Optimizer]=field_default(lambda: Factory(torch.optim.Adam))
    loss_fn_factory : Callable[..., torch.nn.Module] = field_default(lambda: Factory(torch.nn.CrossEntropyLoss, ignore_index=-1))
    loss_wrapper_factory : Callable[..., torch.autograd.Function | None] = field_default(lambda: Factory())
    scheduler_config:scheduler.LRSchedulerConfig | None=None
    lightning_trainer_factory:Callable[..., lightning.Trainer]=field_default(lambda: Factory(lightning.Trainer, precision=32))
    fit_factory:Callable=field_default(lambda: Factory())
    metric_factories:dict[str, Callable[..., metrics.IMetric]]=field_default(lambda: {'loss':Factory(metrics.Loss), 'acc':Factory(metrics.Accuracy)})
    ckpt_path: str | None = None

    def train(self, cfg : cli.ConfigBase):
        assert(isinstance(self.lightning_trainer_factory, Factory))

        if cfg.seed_everything is not None:
            lightning.seed_everything(cfg.seed_everything)

        torch.backends.cudnn.benchmark = self.lightning_trainer_factory['precision'] == "fp32"
        torch.backends.cudnn.enabled = self.lightning_trainer_factory['precision'] == "fp32"

        lightning_model = CoreLightningModel(
            model_factory=cfg.model_factory, 
            optimizer_factory=self.optimizer_factory, 
            loss_fn_factory=self.loss_fn_factory,
            loss_wrapper_factory=self.loss_wrapper_factory,
            metric_factories=self.metric_factories,
            scheduler_config=self.scheduler_config,
        )

        if self.datamodule_factory is not None:
            datamodule = self.datamodule_factory()
            train_loader = datamodule.train_dataloader()
            val_loader = datamodule.val_dataloader()
        else:
            train_dataset : torch.utils.data.Dataset = self.train_dataset_factory()
            val_dataset : torch.utils.data.Dataset = self.val_dataset_factory()

            # FIXME - deal with collate_fn elsewhere
            train_loader : torch.utils.data.DataLoader = self.train_dataloader_factory(dataset = train_dataset, collate_fn=collate_target_tokens_offset_by_one)
            val_loader : torch.utils.data.DataLoader = self.val_dataloader_factory(dataset = val_dataset, collate_fn=collate_target_tokens_offset_by_one)

        # test model on one batch first so we get good errors quickly even when compiling or logging into wandb
        if cfg.pretest and (cfg.compile or len(self.lightning_trainer_factory['logger']) > 0):
            print("Pre-testing model...")
            with torch.no_grad():
                for pretest_batch in train_loader:
                    # if torch.cuda.is_available():
                    #    model = model.to(torch.device('cuda'))
                    #    pretest_batch = pretest_batch.to(torch.device('cuda'))
                    lightning_model.model(pretest_batch[0][0:1,:])
                    break
                print("Testing complete!")

        trainer : lightning.Trainer = self.lightning_trainer_factory(num_sanity_val_steps=0)#, enable_progress_bar=False)#num_sanity_val_steps=1)
        if cfg.compile:
            try:
                lightning_model.model = torch.compile(lightning_model.model)
            except Exception as e:
                print(f"Skipping torch.compile due to error: {e}")

        # #torch._dynamo.config.verbose=True
        trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=self.ckpt_path)

import generator
import torch.amp
from contextlib import nullcontext
class CoreLightningPredictor(cli.IPredictor):
    def __init__(self, cfg : cli.ConfigBase | None, predicting_cfg : Any | None, tokenizer_factory : Callable, checkpoint_path : str | None, seed : int | None = None):
        if seed is not None:
            lightning.seed_everything(seed)

        self.tokenizer = tokenizer_factory()
        if checkpoint_path is None:
            if cfg is not None:
                lightning_model = CoreLightningModel(model_factory=cfg.model_factory)
        else:
            if cfg is not None:            
                lightning_model = CoreLightningModel.load_from_checkpoint(checkpoint_path, model_factory=cfg.model_factory)
            else:
                lightning_model = CoreLightningModel.load_from_checkpoint(checkpoint_path)

        #self.sampler = predicting_cfg.sampler_factory()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
        device_type = 'cuda' if 'cuda' in self.device else 'cpu' # for later use in torch.autocast
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        lightning_model.eval()
        lightning_model.to(self.device)

        self.gen = generator.Generator(lightning_model.model) # FIXME - , self.sampler)

        self.initialized = False
        
    def ingest(self, input_text:str):
        if not self.initialized:
            self.initialized = True
            # special starting token for unconditional generation or regular generation
            start_token_str = self.tokenizer.bos_token
            if start_token_str is None:
                start_token_str = self.tokenizer.eos_token
            input_text = start_token_str + input_text

        # FIXME - move to int16
        tokenized_input_text = torch.LongTensor(self.tokenizer(input_text)['input_ids'], device=self.device).unsqueeze(0)

        with self.ctx:
            self.gen.ingest(tokenized_input_text)

    def predict(self, num_outputs:int):
        with self.ctx:
            for next_token_tensor in self.gen.predict(num_outputs):
                yield self.tokenizer.decode(next_token_tensor[0, ...])
        
    # FIXME - add encode, get_state, set_state

    def reset(self):
        self.initialized = False
        self.reset_encoder()
        self.reset_decoder()

    def reset_decoder(self):
        self.gen.clear_decoder_state()

    def reset_encoder(self):
        self.gen.clear_encoder_state()


def console_clear_last_line():
    print('\033[1A', end='\x1b[2K')

class CoreLightningModel(LightningModule):
    def __init__(
        self,
        model_factory : Callable[..., nn.Module] = Factory(model.core.Decoder),
        optimizer_factory : Callable[..., torch.optim.Optimizer|None] = Factory(),
        loss_fn_factory : Callable[..., nn.Module] = Factory(torch.nn.CrossEntropyLoss, ignore_index=-1),
        loss_wrapper_factory : Callable[..., torch.autograd.Function | None] = Factory(),
        metric_factories : dict[str, Callable[..., metrics.IMetric]] = {'loss':Factory(metrics.Loss), 'acc':Factory(metrics.Accuracy)},
        scheduler_config : scheduler.LRSchedulerConfig | None = None,
    ) -> None:
        super().__init__()        
        # saving additional 'model_str' and 'optimizers_str' since wandb otherwise won't save the full depth of the serialized config, so you can't look back and see all hyperparameters later
        self.save_hyperparameters(dict(model=model_factory, optimizer=optimizer_factory, loss_fn=loss_fn_factory, loss_wrapper=loss_wrapper_factory, scheduler=scheduler_config, model_dict=Factory.toDict(model_factory), optimizer_dict=Factory.toDict(optimizer_factory), loss_fn_dict=Factory.toDict(loss_fn_factory), loss_wrapper_dict=Factory.toDict(loss_wrapper_factory), scheduler_dict=Factory.toDict(scheduler_config), model_str=str(model_factory), optimizer_str=str(optimizer_factory), loss_fn_str=str(loss_fn_factory), loss_wrapper_str=str(loss_wrapper_factory), scheduler_str=str(scheduler_config)))
        #self.logger.experiment.config.update(dict(model=model_factory, optimizers=optimizers_factory))

        self.model = model_factory()
        self.optimizer_factory = optimizer_factory
        self.loss_fn = loss_fn_factory()
        self.loss_wrapper = loss_wrapper_factory()
        self.metrics = dict()
        for name, factory in metric_factories.items():
            self.metrics[name] = factory()
        self.scheduler_config = scheduler_config

        self.tokens_processed = 0
        self.tokens_processed_prev_log = 1
        self.last_iter_time = None
        self.last_log_runtime = 0.0
        self.total_runtime = 0.0
        self.grad_acc_iter = 0

    def on_save_checkpoint(self, checkpoint):
        checkpoint['tokens'] = self.tokens_processed
        checkpoint['runtime'] = self.total_runtime

    def on_load_checkpoint(self, checkpoint):
        self.tokens_processed = checkpoint['tokens']
        self.total_runtime = checkpoint['runtime']

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(x)
    
    def configure_optimizers(self):
        params = [p for _, p in self.named_parameters() if p.requires_grad]

        # if 'weight_decay' in self.optimizers_factory and self.optimizers_factory['weight_decay'] > 0:
        #     # separate out weight decayable parameters
        #     weight_decay = float(self.optimizers_factory['weight_decay'])
        #     params = [
        #         {'params':[p for p in params if p.dim() >= 2], 'weight_decay:': weight_decay},
        #         {'params':[p for p in params if p.dim() < 2], 'weight_decay:': 0}
        #     ]

        optimizer = self.optimizer_factory(params)

        if self.scheduler_config is None:
            return optimizer
        else:
            return dict(optimizer=optimizer, lr_scheduler=self.scheduler_config.to_dict(optimizer))

    def _get_loss_logits_preds(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
    
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.flatten())
        with torch.no_grad():
            preds = logits.argmax(dim=-1)

        return loss, logits, preds

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        loss, logits, preds = self._get_loss_logits_preds(batch, batch_idx)

        margs = metrics.MetricArgs(inputs, logits, preds, labels, loss)
        for metric in self.metrics.values():
            metric.update(margs)

        self.tokens_processed += batch[0].size(-2) * batch[0].size(-1)

        t = time.time()
        if self.last_iter_time is not None:
            dt = t - self.last_iter_time
            self.total_runtime += dt
        else:
            dt = 1.0
        self.last_iter_time = t

        if (batch_idx + 1) % self.trainer.log_every_n_steps == 0:
            ms_since = (self.total_runtime - self.last_log_runtime) * 1000.
            ktok_per_sec = (self.tokens_processed - self.tokens_processed_prev_log) / ms_since
            ms_per = ms_since / self.trainer.log_every_n_steps
            self.last_log_runtime = self.total_runtime
            if self.trainer.is_global_zero:
                if batch_idx > 0 and int(math.log2(self.tokens_processed)) == int(math.log2(self.tokens_processed_prev_log)):
                    console_clear_last_line()
                if torch.cuda.is_available():
                    gb = torch.cuda.memory_allocated(0)/1024/1024/1024.0
                else:
                    gb = 0
    
                str = f"epoch:{self.current_epoch} token:{self.tokens_processed:,} step:{batch_idx} "
                for name, metric in self.metrics.items():
                    metric_value = metric.compute()
                    metric.clear()
                    self.log('train/'+name, metric_value, on_step=True, rank_zero_only=True)
                    str += f'{name}:{metric_value:.4f} '
                str += f"{gb:.1f}gb {int(ms_per)}ms {ktok_per_sec:.2f}kT/s {self.total_runtime:.1f}sec"
                print(str)

                self.tokens_processed_prev_log = self.tokens_processed

            self.log("tokens", float(self.tokens_processed), on_step=True, rank_zero_only=True)
        
        if self.loss_wrapper is not None:
            loss = self.loss_wrapper.apply(loss, logits)

        return loss

    def on_validation_epoch_start(self):
        if self.trainer.is_global_zero:
            print(f"STARTING VALIDATION")
            print()

            # clear metrics
            for metric in self.metrics.values():
                metric.compute()

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        loss, logits, preds = self._get_loss_logits_preds(batch, batch_idx)
        margs = metrics.MetricArgs(inputs, logits, preds, labels, loss)
        for name, metric in self.metrics.items():
            metric.update(margs)
            # on_epoch causes this to be logged in aggregate rather than per batch
            self.log('val/'+name, metric.compute(), on_epoch=True, rank_zero_only=True)
            metric.clear()
        return logits

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            # clear metrics
            for metric in self.metrics.values():
                metric.clear()

            callback_metrics = self.trainer._logger_connector.callback_metrics

            str = f"VALIDATION COMPLETE. "
            for name in self.metrics.keys():
                value = callback_metrics['val/'+name]
                str += f"{value:.4f} "
            console_clear_last_line()
            print(str)
            print()
