import os
import argparse
import torch
import torch.autograd
import torch.utils.data
import lightning
import util.config
from util.config import Factory, RFactory
import lit
from dataclasses import dataclass
import util.logger
from util.logger import log, log_always
from contextlib import nullcontext
from functools import partial

import model.core

import sampler

from dataclasses import dataclass, field

from functools import partial

def factory(*args, **kwargs):
    return field(default_factory=partial(Factory, *args, **kwargs))
def rfactory(*args, **kwargs):
    return field(default_factory=partial(RFactory, *args, **kwargs))

@dataclass
class Config:
    pretest:bool=True
    seed_everything:int|None=1234
    compile:bool=False
    
    model_factory:Factory[model.core.IEncoderDecoder]=None
    optimizer_factory:Factory[torch.optim.Optimizer]=None,
    loss_fn_factory:Factory[torch.nn.Module] = factory(torch.nn.CrossEntropyLoss, ignore_index=-1)
    loss_wrapper_factory:Factory[torch.autograd.Function] = factory()

    tokenizer_factory:Factory=factory()
    dataset_transform_factories:list=factory(list)
    train_dataset_seed:int|None=32
    train_dataset_factory:Factory[torch.utils.data.dataset.Dataset]=None
    train_dataloader_factory:Factory[torch.utils.data.DataLoader]=None
    val_dataset_factory:Factory[torch.utils.data.dataset.Dataset]=None
    val_dataloader_factory:Factory[torch.utils.data.DataLoader]=None
    trainer_factory:Factory[lightning.Trainer]=factory(lightning.Trainer, precision=32)
    fit_factory:Factory=None

    sampler_factory:Factory=factory(sampler.TopKPTailFreeSampler, temperature=1.0, top_p=0.7)

def collate_target_tokens_offset_by_one(batch): 
    tuple_batch = [(d['input_ids'][:-1], d['input_ids'][1:]) for d in batch]
    return torch.utils.data.default_collate(tuple_batch)

def run(command, cfg):
    if cfg.seed_everything is not None:
        lightning.seed_everything(cfg.seed_everything)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = cfg.tokenizer_factory()

    if command == 'train':
        torch.backends.cudnn.benchmark = cfg.trainer_factory['precision'] == "fp32"
        torch.backends.cudnn.enabled = cfg.trainer_factory['precision'] == "fp32"
        block_size = cfg.model_factory['hparams']['block_size']

        train_dataset : torch.utils.data.Dataset = cfg.train_dataset_factory()
        val_dataset : torch.utils.data.Dataset = cfg.val_dataset_factory()

        for xform_factory in cfg.dataset_transform_factories:
            xform = xform_factory(tokenizer=tokenizer, block_size=block_size)
            train_dataset = xform(train_dataset)
            val_dataset = xform(val_dataset)

        if cfg.train_dataset_seed is not None:
            train_dataset = train_dataset.shuffle(seed=cfg.train_dataset_seed)

        val_dataset = val_dataset.take(1024)

        train_loader : torch.utils.data.DataLoader = cfg.train_dataloader_factory(dataset = train_dataset, collate_fn=collate_target_tokens_offset_by_one)
        val_loader : torch.utils.data.DataLoader = cfg.val_dataloader_factory(dataset = val_dataset, collate_fn=collate_target_tokens_offset_by_one)


    if command == 'train':
        model = lit.LightningModel(model_factory=cfg.model_factory, optimizers_factory=cfg.optimizer_factory, loss_fn_factory=cfg.loss_fn_factory, loss_wrapper_factory=cfg.loss_wrapper_factory)

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
            model.model = torch.compile(model.model)
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

        starter_text = "<|endoftext|>In a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
        #starter_text = starter_text + starter_text
        tokenized_starter_text = tokenizer(starter_text)['input_ids']
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
        

        sampler = util.config.sampler_factory()
        print(tokenizer.decode(starter_ids))
        print("...")
        with torch.no_grad():
            with ctx:
                for tok in model.generate_tokens(x, max_new_tokens, sampler, alpha_frequency = 0.25, alpha_presence = 0.25, alpha_decay = 1.0 / 200):
                    print(tokenizer.decode(tok.item()), end='')
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
        try:
            disk_cfg = util.config.eval_first_expr(f.read())
        except util.config.ConfigParseError as e:
            if e.__cause__ is not None:
                raise
            print("Error during configuration parsing:")
            print(e)
            exit()
        cfg = disk_cfg()
        log_always("config:" + "\n" + str(cfg) + "\n")
    errors += util.config.typecheck('cfg', cfg)

    if errors != '':
        print(errors)
        exit()

    run(args.command, cfg)

if __name__ == "__main__":
    cli()
