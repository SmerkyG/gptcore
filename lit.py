from typing import Any, Optional, Tuple

import time
import math

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import torch.optim.lr_scheduler

import metrics

from lightning import LightningModule

from util.config import Factory

def console_clear_last_line():
    print('\033[1A', end='\x1b[2K')

class LightningModel(LightningModule):
    def __init__(
        self,
        model_factory : Factory[nn.Module],
        optimizer_factories : Factory[torch.optim.Optimizer] | list[Factory[torch.optim.Optimizer]] | None,
        loss_fn_factory : Factory[nn.Module],
        loss_wrapper_factory : Factory = Factory(),
        metric_factories : dict[Factory[metrics.IMetric]] = {'loss':Factory(metrics.Loss), 'acc':Factory(metrics.Accuracy)},
        scheduler_factories : Factory[torch.optim.lr_scheduler.LRScheduler] | list[Factory[torch.optim.lr_scheduler.LRScheduler]] | None = None,
    ) -> None:
        super().__init__()        
        # saving additional 'model_str' and 'optimizers_str' since wandb otherwise won't save the full depth of the serialized config, so you can't look back and see all hyperparameters later
        self.save_hyperparameters(dict(model=model_factory, optimizers=optimizer_factories, loss_fn=loss_fn_factory, loss_wrapper=loss_wrapper_factory, schedulers=scheduler_factories, model_str=str(model_factory), optimizers_str=str(optimizer_factories), loss_fn_str=str(loss_fn_factory), loss_wrapper_str=str(loss_wrapper_factory), schedulers_str=str(scheduler_factories)))
        #self.logger.experiment.config.update(dict(model=model_factory, optimizers=optimizers_factory))
        self.model = model_factory()
        self.optimizer_factories = optimizer_factories
        self.scheduler_factories = scheduler_factories
        self.loss_fn = loss_fn_factory()
        self.loss_wrapper = loss_wrapper_factory()
        self.tokens_processed = 0
        self.tokens_processed_prev_log = 1
        self.last_iter_time = None
        self.last_log_runtime = 0.0
        self.total_runtime = 0.0
        self.grad_acc_iter = 0

        self.metrics = dict()
        for name, factory in metric_factories.items():
            self.metrics[name] = factory()

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

        if self.optimizer_factories is None:
            self.optimizer_factories = []
        elif not isinstance(self.optimizer_factories, list):
            self.optimizer_factories = [self.optimizer_factories]
        optimizers = []
        for optimizer_factory in self.optimizer_factories:
            optimizers.append(optimizer_factory(params))

        if self.scheduler_factories is None:
            self.scheduler_factories = []
        elif not isinstance(self.scheduler_factories, list):
            self.scheduler_factories = [self.scheduler_factories]
        schedulers = []
        for scheduler_factory in self.scheduler_factories:
            schedulers.append(scheduler_factory())

        return (optimizers, schedulers)

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
            ms = (self.total_runtime - self.last_log_runtime) * 1000. / self.trainer.log_every_n_steps
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
                str += f"{gb:.1f}gb {ms:.2f}ms {self.total_runtime:.1f}sec"
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
