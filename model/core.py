#from __future__ import annotations

import abc

from util.config import Factory

from typing import Any, Optional, Tuple, List, Iterable, Callable
from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import util.logger as logger
import mask
import norm

from dataclasses import dataclass

import model.interface

from model.hparams import HParams

#import torch._dynamo

class NoOpModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x, *args, **kwargs):
        return x

def defer_module_init(m : nn.Module, init_fn : Callable[[nn.Module], None]):
    def _inner_fn(mm):
        with torch.no_grad(): init_fn(m)
    m.post_init_fn = _inner_fn

class IAttention():
    @abc.abstractmethod
    def forward(self, q:Tensor, k:Tensor, v:Tensor, recurrent_memory : Optional[Tensor] = None):
        raise NotImplementedError

class Attention(nn.Module, IAttention):
    # vanilla attention
    def __init__(self, hparams : HParams, layer_id : int, bias_mask_factory : Factory[mask.IBiasMask] = Factory(mask.CausalBiasMask)): 
        super().__init__(hparams, layer_id)
        assert(layer_id >= 0)
        self.hparams = hparams
        self.bias_mask = bias_mask_factory(hparams, layer_id)

    def forward(self, q:Tensor, k:Tensor, v:Tensor, recurrent_memory : Optional[Tensor] = None):
        return nn.functional.softmax(q @ k.transpose(-2, -1) * q.size(-1)**-0.5 + self.bias_mask, dim=-1) @ v

class LinearAttention(nn.Module, IAttention):
    # unscaled softmax-free attention (unscaled because we're presuming norm will be taken afterwards)
    def __init__(self, hparams : HParams, layer_id : int, mul_mask_factory : Factory[mask.IMulMask] = Factory(mask.CausalMulMask)):
        super().__init__(hparams, layer_id)
        assert(layer_id >= 0)
        T = hparams.block_size
        self.mul_mask = mul_mask_factory(hparams, layer_id)

    def forward(self, q:Tensor, k:Tensor, v:Tensor, recurrent_memory : Optional[Tensor] = None):
        return (q @ k.transpose(-2, -1) * self.mul_mask(q)) @ v

class TorchAttention(nn.Module, IAttention):
    # uses optimized flashattention implementation when possible (as of pytorch2.0.1 this happens only when no mask is specified, but the next version should allow masks too)
    def __init__(self, hparams : HParams = None, layer_id : int = -1, bias_mask_factory : Optional[Factory[mask.IBiasMask]] = None): 
        super().__init__()
        assert(layer_id >= 0)
        self.hparams = hparams
        self.bias_mask = None if bias_mask_factory is None else bias_mask_factory(hparams, layer_id)

    def forward(self, q:Tensor, k:Tensor, v:Tensor, recurrent_memory : Optional[Tensor] = None):
        bias_mask = self.bias_mask(q) if self.bias_mask is not None else None

        return nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias_mask, dropout_p=self.hparams.dropout, is_causal=bias_mask is None)

class TimeLerp(nn.Module):
    def __init__(self, hparams, layer_id:int):
        super().__init__()
        self.hparams = hparams
        self.layer_id = layer_id
        if self.layer_id < self.hparams.n_layer - 1:
            self.offset = nn.ZeroPad2d((0, 0, 1, -1))
            self.mix = nn.Parameter(torch.pow(torch.linspace(0, 1, hparams.d_model), 1.0 - layer_id / hparams.n_layer))

    def forward(self, x):
        if self.layer_id == self.hparams.n_layer - 1:
            return x
        return torch.lerp(x, self.offset(x), self.mix)
        #return x * self.mix + self.offset(x) * (1.0 - self.mix)

class ReluSquared(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.square(torch.relu(x))

class RWKVFeedForwardSubLayer(nn.Module, model.interface.IFeedForwardSubLayer):
    def __init__(self, hparams : HParams, layer_id : int, hidden_activation_factory : Factory = Factory(ReluSquared), gate_activation_factory : Factory = Factory(nn.Sigmoid)):
        super().__init__()
        self.layer_id = layer_id        
        D = hparams.d_model
        F = int(hparams.feedforward_d_model_ratio * hparams.d_model)
        self.time_mixer_hidden = TimeLerp(hparams, layer_id)
        self.time_mixer_gate = TimeLerp(hparams, layer_id)
        self.w_hidden = nn.Linear(D, F, bias=False)
        self.hidden_activation = hidden_activation_factory()
        # FIXME - rename this w_out once we stop using naming for init
        self.w_shrink = nn.Linear(F, D, bias=False)
        self.w_gate = nn.Linear(D, D, bias=False)
        self.gate_activation = gate_activation_factory()

    def forward(self, x : Tensor):
        x_hidden = self.time_mixer_hidden(x)
        x_gate = self.time_mixer_gate(x)
        hidden = self.w_hidden(x_hidden)
        hidden = self.hidden_activation(hidden)
        gate = self.w_gate(x_gate)
        gate = self.gate_activation(gate)
        return gate * self.w_shrink(hidden)
    
class AttentionSubLayer(nn.Module, model.interface.IAttentionSubLayer):
    def __init__(self, hparams : HParams, layer_id : int, 
                 attention_factory : Factory = Factory(TorchAttention), 
                 qkv_norm_factory : Factory = Factory(norm.RMSNorm, weight_scaling=False), 
                 time_mixer_factory = Factory(TimeLerp)):
        super().__init__()
        assert hparams.d_model % hparams.n_head == 0
        self.hparams = hparams

        T = hparams.block_size
        D = hparams.d_model
        H = hparams.n_head
        K = int(hparams.d_qk_ratio * hparams.d_model / hparams.n_head)
        Q = int(hparams.d_qk_ratio * hparams.d_model / hparams.n_head)
        V = int(hparams.d_v_ratio * hparams.d_model / hparams.n_head)
        G = V
        
        self.rotary_positional_embedding = hparams.rotary_positional_embedding_factory(T, Q)

        self.w_query = nn.Linear(D, H * Q, bias=False)
        self.w_key = nn.Linear(D, H * K, bias=False)
        self.w_value = nn.Linear(D, H * V, bias=False)
        self.w_gate = nn.Linear(D, H * G, bias=False)
        self.q_norm = qkv_norm_factory(Q)
        self.k_norm = qkv_norm_factory(K)
        self.v_norm = qkv_norm_factory(V)
        self.w_out = nn.Linear(H * V, D, bias=False)
        def w_out_init(m): m.weight *= ((2 * self.hparams.n_layer)**-0.5)
        defer_module_init(self.w_out, w_out_init)

        self.attention_module = attention_factory(hparams=hparams, layer_id=layer_id)

        self.time_mixer_k = time_mixer_factory(hparams, layer_id)
        self.time_mixer_v = time_mixer_factory(hparams, layer_id)

    def forward(self, xq : Tensor, xk : Tensor, xv : Tensor, recurrent_memory : Optional[Tensor] = None):
        hparams = self.hparams
        B, T, D = xq.size() # batch size, sequence length, token embedding dimension count
        H = hparams.n_head
        K = int(hparams.d_qk_ratio * hparams.d_model / hparams.n_head)
        Q = int(hparams.d_qk_ratio * hparams.d_model / hparams.n_head)
        V = int(hparams.d_v_ratio * hparams.d_model / hparams.n_head)
        G = V

        # Mix xk, xv with the previous timestep in a learned manner to produce new time-mixed xk, xv
        xk = self.time_mixer_k(xk)
        xv = self.time_mixer_v(xv)

        q = self.w_query(xq) # (B, T, H*Q)
        k = self.w_key(xk)# (B, T, H*K)
        v = self.w_value(xv)# (B, T, H*V)
        g = self.w_gate(xq) # (B, T, D) -> (B, T, H*G)

        # transpose H and T dimensions so that all heads can be worked on together with a single matrix (required by all efficient attention/retention implementations)
        q = q.view(B, T, H, Q).transpose(1, 2) # (B, H, T, Q)
        k = k.view(B, T, H, K).transpose(1, 2) # (B, H, T, K)
        v = v.view(B, T, H, V).transpose(1, 2) # (B, H, T, V)

        # normalize each head
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # rotate queries and keys via RoPE / XPos
        q, k = self.rotary_positional_embedding((q, k))

        y = self.attention_module(q, k, v, recurrent_memory) # (B, H, T, V)

        # TransNormer style normalization after multiplying qkv (same as separate groupnorm of each head but w/o a scaling parameter)
        # normalize each head
        y = norm.Norm.F(y)

        # squeeze all the head embeddings together into a single dimension
        y = y.transpose(1, 2).contiguous().view(B, T, H*V) # (B, H, T, V) -> (B, T, H*V)

        # FIXME - putting this AFTER the head recombination made a huge difference! but only when we didn't normalize qkv per head in advance
        #y = norm.Norm.F(y)

        if self.w_gate is not None:
            y = g * y # (B, T, H*V)

        # project the result to our model embedding size
        y = self.w_out(y) # (B, T, H*V) -> (B, T, D)

        y = norm.RMSNorm.F(y) / math.sqrt(2 * self.hparams.n_layer)

        return y

class IResidualOp():
    @abc.abstractmethod
    def forward(self, x : Tensor, sublayer, layer_recurrent_memory : Optional[Tensor] = None):
        raise NotImplementedError

class ResidualAddOp(nn.Module, IResidualOp):
    def __init__(self, hparams : HParams, layer_id : int, sublayer_norm_factory : Factory = Factory(norm.RMSNorm, weight_scaling = False)):
        super().__init__()
        self.dropout = nn.Dropout(hparams.dropout)
        self.norm = sublayer_norm_factory(hparams.d_model)

    def forward(self, x : Tensor, sublayer, layer_recurrent_memory : Optional[Tensor] = None):
        return x + self.dropout(sublayer(self.norm(x)))

class ResidualMixOp(nn.Module, IResidualOp):
    def __init__(self, hparams : HParams, layer_id : int, sublayer_norm_factory : Factory = Factory(norm.RMSNorm, weight_scaling = False)):
        super().__init__()
        self.dropout = nn.Dropout(hparams.dropout)
        self.norm = sublayer_norm_factory(hparams.d_model)
        self.residual_mix = nn.Parameter(torch.ones(1,1,hparams.d_model))

    def forward(self, x : Tensor, sublayer, layer_recurrent_memory : Optional[Tensor] = None):
        return x * self.residual_mix + self.dropout(sublayer(self.norm(x))) * (2 - self.residual_mix)

class TransformerLayer(nn.Module):
    def __init__(self, hparams : HParams, layer_id : int, residual_op_factory : Factory[IResidualOp] = Factory(ResidualMixOp, sublayer_norm_factory = Factory(norm.RMSNorm, weight_scaling = False))):
        super().__init__()
        self.self_attention_sublayer = hparams.self_attention_sublayer_factory(hparams, layer_id)
        self.self_attention_resop = residual_op_factory(hparams, layer_id)

        self.cross_attention_sublayer = hparams.cross_attention_sublayer_factory(hparams, layer_id)
        self.cross_attention_resop = residual_op_factory(hparams, layer_id)

        self.feedforward_sublayer = hparams.feedforward_sublayer_factory(hparams, layer_id)
        self.feedforward_resop = residual_op_factory(hparams, layer_id)

    def forward(self, x : Tensor, encoder_output : Tensor = None, layer_recurrent_memory : Optional[Tensor] = None):
        # self attention (query, key, and value are all based on the same inputs)
        self_attn = lambda y: self.self_attention_sublayer(y, y, y, layer_recurrent_memory)
        x = self.self_attention_resop(x, self_attn) # this code looks a little complicated, but it's just allowing us to swap out whether we add or mix the residual

        # optional cross attention (query is based on the current input, but key and value are based on the encoder's output)
        if encoder_output is not None:
            cross_attn = lambda y: self.cross_attention_sublayer(y, encoder_output, encoder_output, layer_recurrent_memory)
            x = self.cross_attention_resop(x, cross_attn)

        # feedforward network
        feedforward = lambda y: self.feedforward_sublayer(y)
        x = self.feedforward_resop(x, feedforward)

        return x

class Unembedding(nn.Module):
    def __init__(self, weight : Tensor = None):
        super().__init__()
        self.weight = weight

    def forward(self, x : Tensor):
        return F.linear(x, self.weight)

class Transformer(nn.Module):
    def __init__(self, hparams : HParams, 
                 embedding_norm_factory : Factory = Factory(norm.RMSNorm, weight_scaling=False), 
                 positional_embedding_factory : Factory = Factory(NoOpModule),
                 transformer_layer_factory : Factory = Factory(TransformerLayer),
                 share_embedding_weights : bool = True,
                 final_norm_factory : Factory = Factory(norm.RMSNorm, weight_scaling=False),
                 is_decoder : bool = True,
                 unembed_output : bool = True):
        super().__init__()

        self.embed = nn.Embedding(hparams.vocab_size, hparams.d_model)
        def smallInitEmbedding(m): nn.init.uniform_(self.embed.weight, a=-1e-4, b=1e-4)
        defer_module_init(self.embed, smallInitEmbedding) # SmallInit(Emb) per RWKV

        self.embedding_norm = embedding_norm_factory(hparams.d_model)
        self.positional_embedding = positional_embedding_factory(hparams.block_size, hparams.d_model)
        self.embedding_dropout = nn.Dropout(hparams.dropout)

        self.layers = nn.Sequential(*[transformer_layer_factory(hparams, i) for i in range(hparams.n_layer)])

        if share_embedding_weights:
            self.final_norm = NoOpModule()
            self.unembed = Unembedding(self.embed.weight)
        else:
            self.final_norm = final_norm_factory(hparams.d_model)
            self.unembed = nn.Linear(hparams.d_model, hparams.vocab_size, bias = False)
        
        # FIXME - improve weight initialization somehow so it's not external to all the classes
        self._init_weights()

    def forward(self, x : Tensor, encoder_output : Tensor = None, recurrent_memory : Optional[list[Tensor]] = None):
        # convert from input token index to first layer embedding via embedding map
        x = self.embed(x)
        # this extra normalization right after getting the embedding is an improvement from RWKV SmallInit
        # FIXME - instead of normalizing here, can we just multiply by d_model**0.5?
        x = self.embedding_norm(x) # goes from length 1 input to length D^0.5 (768^0.5)
        x = self.positional_embedding(x)
        x = self.embedding_dropout(x)

        # run the main transformer
        # FIXME - annoyingly, we can't just call nn.Sequential because we have to pass through the specific recurrent_memory layer
        for i, layer in enumerate(self.layers):
            x = layer(x, encoder_output, recurrent_memory[i] if recurrent_memory is not None else None)

        # convert from final layer embedding to output logits
        x = self.final_norm(x)
        x = self.unembed(x)

        return x

    def _init_weights(self):
        for name, m in self.named_modules():               
            mean = 0.0
            std = 0.02
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

                nn.init.normal_(m.weight, mean=mean, std=std)
                logger.log(f"{name} mean={mean} std={std}")
            else:
                logger.log(name, "(default init)")
            if getattr(m, 'post_init_fn', None):
                m.post_init_fn(m)
            
class IEncoderDecoder():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def encode(self, x :Tensor):
        raise Exception("Unimplemented")
    def decode(self, x : Tensor):
        raise Exception("Unimplemented")

class Encoder(Transformer, IEncoderDecoder):
    def __init__(self, hparams : HParams):
        super().__init__(hparams, is_decoder=False, unembed_output=True)

    def encode(self, x : Tensor):
        return self.forward(x)

class Decoder(Transformer, IEncoderDecoder):
    def __init__(self, hparams : HParams):
        super().__init__(hparams, is_decoder=True, unembed_output=True)

    def decode(self, x : Tensor, encoder_output : Tensor = None, recurrent_memory : Optional[list[Tensor]] = None):
        return self.forward(x, encoder_output, recurrent_memory)

class EncoderDecoder(nn.Module, IEncoderDecoder):
    def __init__(self, hparams : HParams):
        super().__init__()
        self.encoder = Transformer(hparams, is_decoder=False, unembed_output=False)
        self.decoder = Transformer(hparams, is_decoder=True, unembed_output=True)

    # used for training, not inference
    def forward(self, x : Tensor):
        x = self.encoder(x)
        x = self.decoder(torch.FloatTensor((0,0,0,0)), x)
        return x

    def encode(self, x : Tensor):
        return self.encoder(x)

    def decode(self, x : Tensor, encoder_output : Tensor = None, recurrent_memory : Optional[list[Tensor]] = None):
        return self.decoder(x, encoder_output, recurrent_memory)


