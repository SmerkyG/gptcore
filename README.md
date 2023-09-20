# GPT Core

## Modular code to **create**, **train** cutting edge LLMs ***FAST***

**Crazy Fast:** Pre-train our custom 123M parameter LLM to ~2.8 loss against The Pile on consumer hardware in just **ten minutes** (RTX 4090)

**Batteries included:** Comes with components and full config file setups for SoTA models like RWKV5, LLaMa2, and Hyena

**No Setup:** Automatically stream your dataset from the web, so there's no setup or downloading required

**Research or Learn:** Great for seasoned professionals as well as people new to Machine Learning

**Clean Code:** Best practices and extensible, clearly written, self-documenting code let you focus on your work

- Create, train, and run new LLMs
- Easily compare pre-training results for pre-existing SoTA models like RWKV5, LLaMa2, Hyena or variants
- Put together new models from easy to use components
- Simple Python-syntax model configuration file
- Create new modules and easily run experiments to see which work best
- Read the source code for our custom GPT Alpha LLM to learn top techniques

# TL;DR:

If you want to compare today's open source LLMs, invent new ones, do experiments, or just get a clean framework so you don't have to invent one, this repo might be for you.

## install

```
pip install torch lightning deepspeed einops transformers datasets wandb
```

## configure

If you want to just get started quickly, there are lots of pre-existing examples of config files to look at or try in the `configs/` directory.

GPT Core config files are used to set hyperparameters as well as select the components that make up your model. They follow Python syntax, and ***FULLY SUPPORT AUTOCOMPLETE*** in editors like VSCode.

Example config file:

```
import cli
import model.core
import torch.optim
cli.Config(
    seed_everything = 1337,
    compile = True,
    model_factory = lambda: model.core.Decoder()
    optimizer_factory = lambda: torch.optim.Adam(lr=6e-4),
)
```


GPT Core config files are just a way of constructing objects like you would in Python. The best reference for what options a config file takes is the source code itself. The main outer config definition can be found at the top of `cli.py` - there you will see `@dataclass class Config:` Other classes instantiated within your config file just take whatever arguments their constructors normally would.

The only special thing to know about GPT Core config files is that any parameter that ends in `_factory` should be set via a `lambda:` 

Example: `train_dataloader_factory = lambda: torch.utils.data.DataLoader()`

This is because components require deferred instantiation. Deferral of instantiation is done with a class called `util.config.Factory`, which strongly resembles a Python's `partial` used for partial function invocation. Within a config file, if you specify `lambda:` before you call a function or class constructor, it will automatically wrap your function or class constructor in a `Factory`. This is very similar to how `lambda:` acts in Python already.

## train

```
python cli.py train -c configs/gptalpha.cfg.py
```

GPT Core currently relies on the Lightning trainer, so you can look at the [lightning docs](https://lightning.ai/docs/app/stable/) to learn how to do various tasks like continue from a checkpoint, add custom callbacks during training, etc.

## run a model for inference

```
python cli.py eval -c configs/gptalpha.cfg.py
```
Inference is not the main mission of GPT Core, and will be updated to be more flexible and useful in a future release.
