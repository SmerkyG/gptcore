from typing import Any, Optional, Tuple

import time
import math

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
#import torchmetrics

from lightning import LightningModule

from util.config import Factory

def console_clear_last_line():
    print('\033[1A', end='\x1b[2K')

class Accumulator():
    def __init__(self):
        self.values = []
        self.step = 0
        self.avg = 0.0

    def update(self, value, max_steps):
        # avoid sync point by appending to list and not moving to cpu until necessary
        self.values.append(value.detach())
        if len(self.values) == max_steps:
            # FIXME - sync point here, but is there a way to move these values to cpu faster?
            self.avg = sum(self.values) / len(self.values)
            self.values.clear()

    def is_fresh(self):
        return len(self.values) == 0

    def latest_avg(self):
        return self.avg
        

class LightningModel(LightningModule):
    def __init__(
        self,
        model_factory : Factory[nn.Module],
        optimizers_factory : Factory[torch.optim.Optimizer],
        loss_fn_factory : Factory[nn.Module],
        loss_wrapper_factory : Factory = Factory(),
    ) -> None:
        super().__init__()        
        # saving additional 'model_str' and 'optimizers_str' since wandb otherwise won't save the full depth of the serialized config, so you can't look back and see all hyperparameters later
        self.save_hyperparameters(dict(model=model_factory, optimizers=optimizers_factory, model_str=str(model_factory), optimizers_str=str(optimizers_factory)))
        #self.logger.experiment.config.update(dict(model=model_factory, optimizers=optimizers_factory))
        self.model = model_factory()
        self.optimizers_factory = optimizers_factory
        self.loss_fn = loss_fn_factory()
        self.loss_wrapper = loss_wrapper_factory()
        self.tokens_processed = 0
        self.tokens_processed_prev_log = 1
        self.last_iter_time = None
        self.last_log_runtime = 0.0
        self.logging_loss_accum = Accumulator()
        self.logging_acc_accum = Accumulator()
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
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = [p for _, p in self.named_parameters() if p.requires_grad]

        # if 'weight_decay' in self.optimizers_factory and self.optimizers_factory['weight_decay'] > 0:
        #     # separate out weight decayable parameters
        #     weight_decay = float(self.optimizers_factory['weight_decay'])
        #     params = [
        #         {'params':[p for p in params if p.dim() >= 2], 'weight_decay:': weight_decay},
        #         {'params':[p for p in params if p.dim() < 2], 'weight_decay:': 0}
        #     ]

        return self.optimizers_factory(params)

    def _get_loss_logits_acc(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
    
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.flatten())

        acc = 0.0
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = preds.eq(y).sum() / (y.size(0)*y.size(1))
            # NOTE - torchmetrics.accuracy does NOT WORK at the moment with torch.compile
            #acc = torchmetrics.functional.accuracy(logits.flatten(0, -2), y.flatten(), task="multiclass", num_classes=self.model.hparams.vocab_size)

        return loss, logits, acc

    def training_step(self, batch, batch_idx):
        loss, logits, acc = self._get_loss_logits_acc(batch, batch_idx)
        self.logging_loss_accum.update(loss, self.trainer.log_every_n_steps)
        self.logging_acc_accum.update(acc, self.trainer.log_every_n_steps)

        self.tokens_processed += batch[0].size(-2) * batch[0].size(-1)

        t = time.time()
        if self.last_iter_time is not None:
            dt = t - self.last_iter_time
            self.total_runtime += dt
        else:
            dt = 1.0
        self.last_iter_time = t

        if self.logging_loss_accum.is_fresh():
            ms = (self.total_runtime - self.last_log_runtime) * 1000. / self.trainer.log_every_n_steps
            self.last_log_runtime = self.total_runtime
            if self.trainer.is_global_zero:
                if batch_idx > 0 and int(math.log2(self.tokens_processed)) == int(math.log2(self.tokens_processed_prev_log)):
                    console_clear_last_line()
                if torch.cuda.is_available():
                    gb = torch.cuda.memory_allocated(0)/1024/1024/1024.0
                else:
                    gb = 0
                print(f"token {self.tokens_processed:,}: step {batch_idx} loss {self.logging_loss_accum.latest_avg():.4f}, acc {self.logging_acc_accum.latest_avg()*100:.2f}%, {gb:.1f}gb, {ms:.2f}ms, {self.total_runtime:.1f}sec")
                self.tokens_processed_prev_log = self.tokens_processed

            self.log("train/loss", self.logging_loss_accum.latest_avg(), on_step=True, rank_zero_only=True)
            self.log("train/acc", self.logging_acc_accum.latest_avg(), on_step=True, rank_zero_only=True)
            self.log("tokens", float(self.tokens_processed), on_step=True, rank_zero_only=True)
        
        if self.loss_wrapper is not None:
            loss = self.loss_wrapper.apply(loss, logits)

        return loss

    def on_validation_epoch_start(self):
        if self.trainer.is_global_zero:
            print(f"STARTING VALIDATION")
            print()

    def validation_step(self, batch, batch_idx):
        loss, logits, acc = self._get_loss_logits_acc(batch, batch_idx)
        if self.trainer.is_global_zero:
            self.log('val/loss', loss, on_epoch=True) # on_epoch causes this to be logged in aggregate rather than per batch
            self.log('val/acc', acc, on_epoch=True)
        return logits

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            metrics = self.trainer._logger_connector.callback_metrics
            loss = metrics['val/loss']
            acc = metrics['val/acc']
            console_clear_last_line()
            print(f"VALIDATION COMPLETE. loss:{loss:.2f} acc:{acc*100.0:.2f}% ppl:{math.exp(loss):.2f}")
            print()
