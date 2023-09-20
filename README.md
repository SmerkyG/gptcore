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

## datasets

Consider trying other huggingface datasets, such as OpenWebText:

`train_dataset = datasets.load_dataset(path='Skylion007/openwebtext', streaming=True, split='train')`

## logging

The example config files show how to use [WandB](https://wandb.ai/home) for logging, but any logger, multiple loggers, or even no logger can be used. See [lightning docs](https://lightning.ai/docs/app/stable/) for info.

## compile

Setting `compile = True` causes your model to be compiled using `torch.compile(model)` This can dramatically improve training speeds.

It can also cause very slow startup times and even break, if you use features in your model that are unsupported by dynamo. (for example, the `complex` torch datatype)

## checkpointing and validation splits

To have a validation done and checkpoint written, add the following config setting to trainer:

```
trainer_factory = lambda: lightning.Trainer(
    val_check_interval=1024,
)
```

## fine-tuning or continuing training from a checkpoint

Continue a training run from an existing checkpoint by supplying the following kind of config setting:

```
fit_factory = lambda: lightning.Trainer.fit(
    ckpt_path='checkpoints/epoch=0-step=256.ckpt',
)
```
Unlike many of the factory settings in config which are deferred class instantiations, this lambda represents the deferred function call to your trainer's `fit()` function.

## run a model for inference

```
python cli.py eval -c configs/gptalpha.cfg.py
```
Inference is not the main mission of GPT Core, and will be updated to be more flexible and useful in a future release.

## GPT Alpha

GPT Alpha is the custom LLM that comes with GPT Core. It employs many components found in state of the art LLM research, as well as our own experiments to achieve fast training to low perplexity and loss.

The following are some of the improvements it uses:

### Small embedding initializations
Embeddings are initialized to a small value but are then immediately normalized. This immediately creates an embedding space that is well distributed around the unit sphere, and converges in a rapid fashion with desirable qualities.
### Weight Tying
The unembedding which translates from the final layer embeddings back to token ids relies on the same weights as the embedding, resulting in faster training and significantly smaller model size.
### Rotary Positional Embedding (RoPE) / Adjustment for Linear Biases (ALiBi)
We use RoPE or ALiBi to adjust query and key values before the attention computation, so that a flexible form of positional information is used by the network. This causes faster learning as well as
Unfortunately, we do not yet include the upcoming version of FlashAttention that would support ALiBi style biases, so ALiBi is currently significantly slower than RoPE using our platform.
### RMSNorm
Prior formulations used BatchNorm or LayerNorm, but with proper initialization and scaling we can use root mean square norm with no weight learning. This results in faster learning with no downsides.
### Attention Sublayer Gating
A learned gate is added to the final output of the attention sublayer.
### Query, Key, and Value Normalization
We normalize query, keys, and values prior to the computation of attention.
### Value Head Size Ratio
We allow values to be larger than keys and queries. This lets the network "think harder" about the embeddings it attends to, with only a small loss in training speed.
### Attention Group Normalization
We normalize each head of the attention computation output separately using RMSNorm to form a kind of group norm by head.
### Time Lerp
Instead of using the token embedding at a specific sequence position, we use a mix of it with the sequentially prior embedding. This costs very little but dramatically improves network performance.
### Specialized Feed Forward Network
We adopt most of the practices from the RWKV channel mix feed forward network. This includes Time Lerp, which is used in our feed forward network as well as other places.
### Residual Mixing
Sublayers (attention and feed forward) are mixed together with the residual using a specific kind of learned ratio. We find this has slightly better results and costs very little to evaluate.

## Roadmap

- colab notebook so anyone can easily try GPT Core
- documentation on components
- documentation on dataset collation etc.
- true support for recurrent state
- encoder-decoder models and components (maybe T5?)
- some wandb charts of actual training runs
- clear and easy mechanisms for distributed training
- self-implemented modular components that match parts of third party models like retnet, rwkv5
- improved inference
- allow separation of config for model, logging, dataset
