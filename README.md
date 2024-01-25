# GPT Core

## ***FAST*** Modular code to **create** and **train** cutting edge LLMs

**Crazy Fast:** Pre-train our custom 123M parameter LLM to ~3.5 validation loss against The Pile in just **twenty minutes** on a consumer grade GeForce RTX™ 4090

**Batteries included:** Comes with components and full config file setups for state of the art models like RWKV5, LLaMa2, and Hyena

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

If you want to compare training today's open source LLMs, invent new ones, do experiments, or just get a clean framework so you don't have to invent one, this repo might be for you.

## install

```
pip install torch lightning deepspeed einops transformers datasets wandb
```

## configure

If you want to just get started quickly, there are lots of pre-existing examples of config files to look at or try in the `configs/` directory.

GPT Core config files are used to set hyperparameters as well as select the components that make up your model. They follow Python syntax, and ***FULLY SUPPORT AUTOCOMPLETE*** in editors like VSCode. This makes it very easy to edit them, check them for errors, or see what options are available - all from within your favorite IDE.

Example config file:

```python
import cli
import model.core
import lit
import torch.optim

MAX_SEQUENCE_LENGTH = 1024

cli.Config(
    seed_everything = 1337,
    compile = True,
    model_factory = lambda: model.core.Decoder(
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    ),
    trainer_factory = lambda: lit.CoreLightningTrainer(
        optimizer_factory = lambda: torch.optim.Adam(lr=6e-4),
    ),
)
```

GPT Core config files contain imports, then optional variable assignments, then a final expression that is considered the result of the config file.

These files are just a way of constructing an object like you would in Python. The best reference for what options a configuration object takes is the source code itself. The main outer config definition can be found at the top of `cli.py` - there you will see `@dataclass class Config:` Other classes instantiated within your config file just take whatever arguments their constructors normally would.

The only special thing to know about GPT Core config files is that any parameter that ends in `_factory` should be set via a `lambda:` 

Example: `train_dataloader_factory = lambda: torch.utils.data.DataLoader()`

This is because components require deferred instantiation. Deferral of instantiation is done with a class called `util.config.Factory`, which strongly resembles a Python's `partial` used for partial function invocation. Within a config file, if you specify `lambda:` before you call a function or class constructor, it will automatically wrap your function or class constructor in a `Factory`. This is very similar to how `lambda:` acts in Python already.

## train

```bash
python cli.py train -c configs/gptalpha.cfg.py
```

GPT Core currently supports the Lightning trainer via its class `lit.CoreLightningTrainer`. The GPT Core class `lit.CoreLightningTrainer` exactly matches the Lightning API, slightly flattened for ease of use in config files. So as you explore the autocomplete for CoreLightningTrainer you can look at the [lightning docs](https://lightning.ai/docs/app/stable/) to learn how to do various tasks like continue from a checkpoint, add custom callbacks during training, etc. 

## datasets

Consider trying other huggingface datasets, such as OpenWebText:

`train_dataset = datasets.load_dataset(path='Skylion007/openwebtext', streaming=True, split='train')`

## logging

The example config files show how to use [WandB](https://wandb.ai/home) for logging, but any logger, multiple loggers, or even no logger can be used. See [lightning docs](https://lightning.ai/docs/app/stable/) for more info.

You can easily choose which metrics to log using the config system and parameterize them like so (or create new ones):
```python
metric_factories=dict(
    loss=lambda: metrics.Loss(),
    acc=lambda: metrics.Accuracy()
),
```

## compile

Setting `compile = True` causes your model to be compiled using `torch.compile(model)` This can dramatically improve training speeds.

It can also cause very slow startup times and even break, if you use features in your model that are unsupported by dynamo. (for example, the `complex` torch datatype)

## checkpointing and validation splits

To have a validation done and checkpoint written, add the following config setting to trainer:

```python
trainer_factory = lambda: lightning.Trainer(
    val_check_interval=1024, # choose whatever number of steps you'd like here
)
```

## fine-tuning or continuing training from a checkpoint

Continue a training run from an existing checkpoint by supplying the following kind of config setting:

```python
trainer_factory = lambda: lit.CoreLightningTrainer(
    ckpt_path='checkpoints/epoch=0-step=256.ckpt',
)
```

## run a model for inference

```bash
python cli.py eval -c configs/gptalpha.cfg.py
```
Inference is not the main mission of GPT Core, and will be updated to be more flexible and useful in a future release.

## GPT Alpha

GPT Alpha is the custom LLM that comes with GPT Core. It employs many of the best components found in state of the art LLM research, as well as our own experiments to achieve fast training to low perplexity and loss.

The following are some of the improvements it uses:

#### Small embedding initializations [citation](https://github.com/BlinkDL/SmallInitEmb)
Embeddings are initialized to a small value but are then immediately normalized. This immediately creates an embedding space that is well distributed around the unit hypersphere, and converges in a rapid fashion with desirable qualities, even with no warmup.
#### Weight Tying [citation](https://arxiv.org/abs/1608.05859v3)
The unembedding which translates from the final layer embeddings back to token ids relies on the same weights as the embedding, resulting in faster training and significantly smaller model size.
#### Attention with Linear Biases (ALiBi) [citation](https://arxiv.org/abs/2108.12409)
We use ALiBi to bias attention results, so that a flexible form of positional information is used by the network. 
#### RMSNorm / L2Norm [citation](https://arxiv.org/abs/1910.07467) [citation](https://arxiv.org/abs/2307.14995)
Prior formulations used BatchNorm or LayerNorm, but with proper initialization and scaling we can use root mean square norm and sometimes unscaled L2 norm with no weight learning. This results in faster learning with no downsides.
#### Attention Sublayer Gating [citation](https://arxiv.org/abs/1804.03999v3)
A learned gate is added to the final output of the attention sublayer.
#### Query, Key, and Value Normalization [self-citation](https://github.com/SmerkyG/gptcore)
We normalize query, keys, and values prior to the computation of attention.
#### Independent Value Head Size [citation](https://arxiv.org/abs/2307.08621)
We allow values to be larger than keys and queries. This lets the network "think harder" about the embeddings it attends to, with only a small loss in training speed.
#### Attention Group Normalization [citation](https://arxiv.org/abs/2307.08621)
We normalize each head of the attention computation output separately using RMSNorm to form a kind of group norm by head.
#### Time Lerp [citation](https://arxiv.org/abs/2305.13048)
Instead of using the token embedding at a specific sequence position, we use a mix of it with the sequentially prior embedding. This costs very little but dramatically improves network performance.
#### Specialized Feed Forward Network [citation](https://arxiv.org/abs/2305.13048) [citation](https://arxiv.org/abs/2002.05202) [citation](https://arxiv.org/abs/2109.08668)
We adopt most of the practices from the RWKV channel mix feed forward network. This includes Time Lerp, which is used in our feed forward network as well as other places, and specialized gating.
#### Residual Mixing [self-citation](https://github.com/SmerkyG/gptcore)
Sublayers (attention and feed forward) are mixed with the residual using a learned vector ratio L via `L*residual+(2-L)*value`. We find this has slightly better results and costs very little to evaluate.
#### No bias [citation](https://arxiv.org/abs/2212.14034)
Linear layers contain only weights without biases throughout the model
#### Sequence Packing [citation](https://arxiv.org/abs/1910.10683)
Sequences in the training data that do not reach the end of the sequence length buffer are packed together instead of padded to the end, so that every token examined is useful for learning instead of large numbers of padding tokens becoming a waste of compute.

## Roadmap

- colab notebook so anyone can easily try GPT Core
- documentation on components
- documentation on dataset collation etc.
- improved weight initialization mechanisms
- true support for recurrent state
- encoder-decoder models and components (maybe T5?)
- some wandb charts of actual training runs
- clear and easy mechanisms for distributed training
- self-implemented modular components that match parts of third party models like retnet, rwkv5
- improved inference
- allow separation of config for model, logging, dataset
- testing apparatus (BLEU score etc.)
- Mixture of Experts

#### Possible future additions to GPT Alpha:
- MixCE
