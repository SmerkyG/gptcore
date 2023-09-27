import os
import argparse
import torch
import torch.autograd
import torch.amp
import torch.backends.cuda
import torch.backends.cudnn
import torch.utils.data.dataset
import lightning
import util.config
from util.config import Factory
import lit
from dataclasses import dataclass
import util.logger
from util.logger import log, log_always
from contextlib import nullcontext
from functools import partial

import model.core

import sampler
import scheduler

import metrics

import generator

from dataclasses import dataclass, field

from functools import partial

from typing import Callable, Any

import cfgctx
import dataset.tokenizer
import dataclasses

def factory(*args, **kwargs):
    return field(default_factory=partial(Factory, *args, **kwargs))


@dataclass
class Config:
    batch_size:int
    block_size:int

    pretest:bool=True
    seed_everything:int|None=1234
    compile:bool=False
    
    lightning_model_factory:Callable[..., lit.LightningModel]=Factory(lit.LightningModel)

    tokenizer_factory:Callable=factory()
    #train_dataset_seed:int|None=32
    train_dataset_factory:Callable[..., torch.utils.data.dataset.Dataset]=None
    train_dataloader_factory:Callable[..., torch.utils.data.DataLoader]=None
    val_dataset_factory:Callable[..., torch.utils.data.dataset.Dataset]=None
    val_dataloader_factory:Callable[..., torch.utils.data.DataLoader]=None
    trainer_factory:Callable[..., lightning.Trainer]=factory(lightning.Trainer, precision=32)
    fit_factory:Callable=None
    metric_factories:dict[str, Callable[..., metrics.IMetric]]=field(default_factory=lambda:{'loss':Factory(metrics.Loss), 'acc':Factory(metrics.Accuracy)})

    sampler_factory:Factory=factory(sampler.TopKPTailFreeSampler, temperature=1.0, top_p=0.7)

def collate_target_tokens_offset_by_one(batch): 
    values = torch.utils.data.default_collate(batch)
    return values[..., :-1], values[..., 1:]

import dataset

def run(command, cfg : Config):
    if cfg.seed_everything is not None:
        lightning.seed_everything(cfg.seed_everything)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cfgctx.batch_size = cfg.batch_size
    cfgctx.block_size = cfg.block_size
    cfgctx.tokenizer = cfg.tokenizer_factory()

    # NOTE - we have to replace identifier accessors here, because if dataloader forks new processes, then
    #  those won't be able to access the values from the loaded cfgctx module unless they're materialized in advance here
    for field in dataclasses.fields(cfg):
        setattr(cfg, field.name, util.config.recursively_replace_identifier_accessors(getattr(cfg, field.name)))

    log_always("config:" + "\n" + str(cfg) + "\n")

    if command == 'train':
        torch.backends.cudnn.benchmark = cfg.trainer_factory['precision'] == "fp32"
        torch.backends.cudnn.enabled = cfg.trainer_factory['precision'] == "fp32"

        model = cfg.lightning_model_factory()


        train_dataset : torch.utils.data.Dataset = cfg.train_dataset_factory()
        val_dataset : torch.utils.data.Dataset = cfg.val_dataset_factory()

        train_loader : torch.utils.data.DataLoader = cfg.train_dataloader_factory(dataset = train_dataset, batch_size = cfgctx.batch_size, collate_fn=collate_target_tokens_offset_by_one)
        val_loader : torch.utils.data.DataLoader = cfg.val_dataloader_factory(dataset = val_dataset, batch_size = cfgctx.batch_size, collate_fn=collate_target_tokens_offset_by_one)

        # test model on one batch first so we get good errors quickly even when compiling or logging into wandb
        if cfg.pretest and (cfg.compile or len(cfg.trainer_factory['logger']) > 0):
            print("Pre-testing model...")
            with torch.no_grad():
                for pretest_batch in train_loader:
                    # if torch.cuda.is_available():
                    #    model = model.to(torch.device('cuda'))
                    #    pretest_batch = pretest_batch.to(torch.device('cuda'))
                    model.model(pretest_batch[0][0:1,:])
                    break
                print("Testing complete!")

        trainer : lightning.Trainer = cfg.trainer_factory(num_sanity_val_steps=0)#, enable_progress_bar=False)#num_sanity_val_steps=1)
        if cfg.compile:
            try:
                model.model = torch.compile(model.model)
            except Exception as e:
                print(f"Skipping torch.compile due to error: {e}")

        # #torch._dynamo.config.verbose=True
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, **(cfg.fit_factory.kwargs))

    if command == 'eval':
        max_new_tokens = 500
        device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
        device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        out_dir = 'out'
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        hparams = checkpoint['hparams']
        checkpoint = torch.load(ckpt_path, map_location=device)
        model = model.core.Decoder(hparams)
        state_dict = checkpoint['model']
        unwanted_prefix = 'model._orig_mod.'
        wanted_prefix = ''
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[wanted_prefix + k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        gen = generator.Generator(model)

        starter_text = "<|endoftext|>In a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
        #starter_text = starter_text + starter_text
        tokenized_starter_text = cfgctx.tokenizer(starter_text)['input_ids']
        starter_ids = tokenized_starter_text[-1025:-1]
        predicted = tokenized_starter_text[-1024:]
        x = (torch.tensor(starter_ids, dtype=torch.long, device=device)[None, ...])
        y = (torch.tensor(predicted, dtype=torch.long, device=device)[None, ...])

        # with torch.no_grad():
        #     logits = model.forward(x)
        #     predicted_labels = logits.argmax(dim=-1)
        #     acc = predicted_labels.eq(y).sum() / float(y.size(0)*y.size(1))
        #     print(f"acc {float(acc)}")

        #     print(tokenizer.decode(predicted_labels.squeeze(0).tolist()))
        

        sampler = cfg.sampler_factory()
        print(cfgctx.tokenizer.decode(starter_ids))
        print("...")
        with torch.no_grad():
            with ctx:
                for tok in gen.generate_tokens(x, max_new_tokens, sampler, alpha_frequency = 0.25, alpha_presence = 0.25, alpha_decay = 1.0 / 200):
                    print(cfgctx.tokenizer.decode(tok.item()), end='')
                    #print(decode(y[0].tolist()))
                print('')
                print('---------------')

def cli():
    parser = argparse.ArgumentParser(description='train and execute pytorch models using lightning', add_help=True)
    parser.add_argument('command', type=str, nargs='?', choices=['train', 'eval'])
    parser.add_argument('-c', '--config', type=str, required=True, help='path to configuration file')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='show more model info at startup')

    #parser.cfg = cfg
    args = parser.parse_args()

    if args.verbose:
        util.logger.Logger.log_level = 1

    if args.command is None:
        parser.print_usage();
        return

    errors = ''

    with open(args.config, mode="rt", encoding="utf-8") as f:
        disk_cfg_str = f.read()
        try:
            disk_cfg = util.config.eval_first_expr(disk_cfg_str)
        except util.config.ConfigParseError as e:
            if e.__cause__ or e.__context__ is not None:
                raise
            print("Error during configuration parsing:")
            print(e)
            exit()

    errors += util.config.typecheck('cfg', disk_cfg, required_type=Config)
    if errors != '':
        print(errors)
        exit()
    try:
        cfg = disk_cfg()
    except TypeError:
        raise util.config.ConfigInstantiationError("Error instantiating config - did you forget a 'lambda:', causing a class or function to be called immediately with not all of its arguments supplied? Unfortunately we can't know where in the config... see above exception for type involved")

    run(args.command, cfg)

if __name__ == "__main__":
    cli()
