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

from dataclasses import dataclass, field

from typing import Callable, Any, Generator

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
        raise NotImplementedError()

class IPredictor:
    def ingest(self, input_text:str) -> None:
        raise NotImplementedError()
    def predict(self, num_outputs:int) -> Generator[str, None, None]:
        raise NotImplementedError()
    # FIXME - add encode, get_state, set_state
    def reset(self):
        raise NotImplementedError()
    def reset_encoder(self):
        raise NotImplementedError()
    def reset_decoder(self):
        raise NotImplementedError()


@dataclass
class Config(ConfigBase):
    trainer_factory:Callable[..., ITrainer]=field_default(lambda: Factory())
    predictor_factory:Callable[..., IPredictor]=field_default(lambda: Factory())

def cli():
    parser = argparse.ArgumentParser(description='train and execute pytorch models using lightning', add_help=True)
    parser.add_argument('command', type=str, nargs='?', choices=['train', 'eval'])
    parser.add_argument('-c', '--config', type=str, required=True, help='path to configuration file')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='show more model info at startup')
    parser.add_argument('-s', '--set', metavar='NAME=value', type=str, nargs='+', help='set global config value(s) in python syntax e.g. NAME=\'John\'')

    #parser.cfg = cfg
    args = parser.parse_args()

    if args.verbose:
        util.logger.Logger.log_level = 1

    if args.command is None:
        parser.print_usage();
        return

    errors = ''

    macros = {}
    if args.set is not None:
        for macro_str in args.set:
            parts = macro_str.split("=")
            if len(parts) != 2 or len(parts[0])==0 or len(parts[1])==0:
                print(f'commandline argument not specified correctly e.g. -s NAME=\'John\'\nGot: {macro_str}')
                return
            macros[parts[0]] = parts[1]

    with open(args.config, mode="rt", encoding="utf-8") as f:
        disk_cfg_str = f.read()
        try:
            disk_cfg = util.config.eval_first_expr(disk_cfg_str, macros)
        except util.config.ConfigParseError as e:
            if e.__cause__ or e.__context__ is not None:
                raise
            print("Error during configuration parsing:")
            print(e)
            return

    errors += util.config.typecheck('cfg', disk_cfg, required_type=Config)
    if errors != '':
        print(errors)
        return
    try:
        cfg : Config = disk_cfg()
    except TypeError:
        raise util.config.ConfigInstantiationError("Error instantiating config - did you forget a 'lambda:', causing a class or function to be called immediately with not all of its arguments supplied? Unfortunately we can't know where in the config... see above exception for type involved")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # NOTE - we have to replace identifier accessors here, because if dataloader forks new processes, then
    #  those won't be able to access the values from the loaded cfgctx module unless they're materialized in advance here
    for field in dataclasses.fields(cfg):
        setattr(cfg, field.name, util.config.recursively_replace_identifier_accessors(getattr(cfg, field.name)))

    if args.command == 'train':
        log_always("config:" + "\n" + str(cfg) + "\n")

        trainer = cfg.trainer_factory()
        trainer.train(cfg)

    import sys    
    if args.command == 'eval':
        predictor = cfg.predictor_factory(cfg=cfg)

        def console_clear_last_line():
            print('\033[1A', end='\x1b[2K')

        n_tokens = 128
        while True:
            print("Enter text to ingest, followed by Ctrl-D then Enter.")
            text = ""
            while True:
                b = sys.stdin.buffer.readline()
                s = str(b, 'UTF-8')
                eof = s.find('\x04')
                if eof >= 0:
                    text += s[:eof]
                    break
                text += s
            try:
                predictor.ingest(text)
            except EOFError:
                pass
            except Exception as e:
                print("Error:", e)
                print("Resetting...")
                predictor.reset()
                continue

            while True:
                print()
                line = input(f"Commands: [Enter] predict {n_tokens} tokens, [num_tokens] to predict, [i]ngest, [r]eset: ")
                if len(line) > 0:
                    if line[0].lower() == 'r':
                        print("Resetting...")
                        predictor.reset()
                        break
                    if line[0].lower() == 'i':
                        break
                    new_n_tokens = int(line)
                    if new_n_tokens > 0:
                        n_tokens = new_n_tokens
                console_clear_last_line()
                try:
                    for next_token_str in predictor.predict(num_outputs=n_tokens):
                        print(next_token_str, end='')
                except Exception as e:
                    print("Error:", e)
                    print("Resetting...")
                    predictor.reset()
                    break


if __name__ == "__main__":
    cli()
