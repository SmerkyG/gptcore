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
from dataclasses import dataclass
import util.logger
from util.logger import log, log_always
from contextlib import nullcontext
from functools import partial

from dataclasses import dataclass, field

from functools import partial

from typing import Callable, Any

import cfgctx
import dataclasses

def field_default(fn):
    return field(default_factory=fn)

@dataclass
class ConfigBase:
    initializers:Any = None

    pretest:bool=True
    seed_everything:int|None=1234
    compile:bool=False

    model_factory:Callable[..., torch.nn.Module]=field_default(lambda: Factory(torch.nn.Module))

class ITrainer:
    def train(self, cfg : ConfigBase):
        pass

class IEvaluator:
    def eval(self, cfg : ConfigBase):
        pass

@dataclass
class Config(ConfigBase):
    trainer_factory:Callable[..., ITrainer]=field_default(lambda: Factory())
    evaluator_factory:Callable[..., IEvaluator]=field_default(lambda: Factory())

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
        cfg : Config = disk_cfg()
    except TypeError:
        raise util.config.ConfigInstantiationError("Error instantiating config - did you forget a 'lambda:', causing a class or function to be called immediately with not all of its arguments supplied? Unfortunately we can't know where in the config... see above exception for type involved")

    if cfg.seed_everything is not None:
        lightning.seed_everything(cfg.seed_everything)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # cfgctx.batch_size = cfg.batch_size
    # cfgctx.block_size = cfg.block_size
    # cfgctx.tokenizer = cfg.tokenizer_factory()

    # NOTE - we have to replace identifier accessors here, because if dataloader forks new processes, then
    #  those won't be able to access the values from the loaded cfgctx module unless they're materialized in advance here
    for field in dataclasses.fields(cfg):
        setattr(cfg, field.name, util.config.recursively_replace_identifier_accessors(getattr(cfg, field.name)))

    log_always("config:" + "\n" + str(cfg) + "\n")

    if args.command == 'train':
        trainer = cfg.trainer_factory()
        trainer.train(cfg)

    if args.command == 'eval':
        evaluator = cfg.evaluator_factory()
        evaluator.eval(cfg)

if __name__ == "__main__":
    cli()
