#from __future__ import annotations

import abc

from util.config import Factory

from typing import Callable, Any, Optional, Tuple, List, Iterable
from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import util.logger as logger
import mask
import norm

import model.interface
import posemb.interface

from model.hparams import HParams

#import torch._dynamo

def defer_module_init(m : nn.Module, init_fn : Callable[[nn.Module], None]):
    def _inner_fn(mm):
        with torch.no_grad(): init_fn(m)
    m.post_init_fn = _inner_fn

class IAttention():
    @abc.abstractmethod
    def forward(self, q:Tensor, k:Tensor, v:Tensor, recurrent_memory : Optional[Tensor] = None):
        raise NotImplementedError
    
    @abc.abstractmethod
    def __call__(self, q:Tensor, k:Tensor, v:Tensor, recurrent_memory : Optional[Tensor] = None):
        raise NotImplementedError
        

class TransformerLayerPart(nn.Module):
    hparams = None
    layer_id = None
    def __init__(self):
        super().__init__()
        self.hparams : HParams = TransformerLayerPart.hparams
        self.layer_id : int = TransformerLayerPart.layer_id

class Attention(TransformerLayerPart, IAttention):
    # vanilla attention
    def __init__(self, bias_mask_factory : Callable[..., mask.IBiasMask] = Factory(mask.CausalBiasMask)): 
        super().__init__()
        self.bias_mask = bias_mask_factory(self.hparams.max_sequence_length, self.hparams.n_head, self.layer_id)

    def forward(self, q:Tensor, k:Tensor, v:Tensor, recurrent_memory : Optional[Tensor] = None):
        return nn.functional.softmax(q @ k.transpose(-2, -1) * q.size(-1)**-0.5 + self.bias_mask(q), dim=-1) @ v

class LinearAttention(TransformerLayerPart, IAttention):
    # unscaled softmax-free attention (unscaled because we're presuming norm will be taken afterwards)
    def __init__(self, mul_mask_factory : Callable[..., mask.IMulMask] = Factory(mask.CausalMulMask)):
        super().__init__()
        self.mul_mask = mul_mask_factory(self.hparams.max_sequence_length, self.hparams.n_head, self.layer_id)

    def forward(self, q:Tensor, k:Tensor, v:Tensor, recurrent_memory : Optional[Tensor] = None):
        return (q @ k.transpose(-2, -1) * self.mul_mask(q)) @ v

class TorchAttention(TransformerLayerPart, IAttention):
    # uses optimized flashattention implementation when possible (as of pytorch2.0.1 this happens only when no mask is specified, but the next version should allow masks too)
    def __init__(self, bias_mask_factory : Callable[..., mask.IBiasMask] | None = None): 
        super().__init__()
        self.bias_mask = None if bias_mask_factory is None else bias_mask_factory(self.hparams.max_sequence_length, self.hparams.n_head, self.layer_id)

    def forward(self, q:Tensor, k:Tensor, v:Tensor, recurrent_memory : Optional[Tensor] = None):
        bias_mask = self.bias_mask(q) if self.bias_mask is not None else None

        return nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias_mask, dropout_p=self.hparams.dropout, is_causal=bias_mask is None)

class TimeLerp(TransformerLayerPart):
    def __init__(self):
        super().__init__()
        if self.layer_id < self.hparams.n_layer - 1:
            self.offset = nn.ZeroPad2d((0, 0, 1, -1))
            self.mix = nn.Parameter(torch.pow(torch.linspace(0, 1, self.hparams.d_model), 1.0 - self.layer_id / self.hparams.n_layer))

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

class RWKVFeedForwardSubLayer(TransformerLayerPart, model.interface.IFeedForwardSubLayer):
    def __init__(self, hidden_activation_factory : Callable = Factory(ReluSquared), gate_activation_factory : Callable = Factory(nn.Sigmoid)):
        super().__init__()
        self.layer_id = self.layer_id        
        D = self.hparams.d_model
        F = int(self.hparams.feedforward_d_model_ratio * self.hparams.d_model)
        self.time_mixer_hidden = TimeLerp()
        self.time_mixer_gate = TimeLerp()
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

class AttentionSubLayer(TransformerLayerPart, model.interface.IAttentionSubLayer):
    def __init__(self,
                 attention_factory : Callable[..., IAttention] = Factory(TorchAttention), 
                 qkv_norm_factory : Callable = Factory(norm.RMSNorm, weight_scaling=False), 
                 time_mixer_factory : Callable = Factory(TimeLerp)):
        super().__init__()
        hparams = self.hparams
        layer_id = self.layer_id
        assert hparams.d_model % hparams.n_head == 0

        T = hparams.max_sequence_length
        D = hparams.d_model
        H = hparams.n_head
        KVH = int(hparams.n_head * hparams.n_kv_head_ratio)
        K = int(hparams.d_qk_ratio * hparams.d_model / hparams.n_head)
        Q = int(hparams.d_qk_ratio * hparams.d_model / hparams.n_head)
        V = int(hparams.d_v_ratio * hparams.d_model / hparams.n_head)
        G = V
        
        self.rotary_positional_embedding = hparams.rotary_positional_embedding_factory(T, Q)

        self.w_query = nn.Linear(D, H * Q, bias=False)
        self.w_key = nn.Linear(D, KVH * K, bias=False)
        self.w_value = nn.Linear(D, KVH * V, bias=False)
        self.w_gate = nn.Linear(D, H * G, bias=False)
        self.q_norm = qkv_norm_factory(Q)
        self.k_norm = qkv_norm_factory(K)
        self.v_norm = qkv_norm_factory(V)
        self.w_out = nn.Linear(H * V, D, bias=False)
        def w_out_init(m): m.weight *= ((2 * self.hparams.n_layer)**-0.5)
        defer_module_init(self.w_out, w_out_init)

        self.attention_module = attention_factory()

        self.time_mixer_k = time_mixer_factory()
        self.time_mixer_v = time_mixer_factory()

    def forward(self, xq : Tensor, xk : Tensor, xv : Tensor, recurrent_memory : Optional[Tensor] = None):
        hparams = self.hparams
        B, T, D = xq.size() # batch size, sequence length, token embedding dimension count
        H = hparams.n_head
        KVH = int(hparams.n_head * hparams.n_kv_head_ratio)
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
        k = k.view(B, T, KVH, K).transpose(1, 2) # (B, KVH, T, K)
        v = v.view(B, T, KVH, V).transpose(1, 2) # (B, KVH, T, V)

        # normalize each head
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # rotate queries and keys via RoPE / XPos
        q, k = self.rotary_positional_embedding((q, k))

        # support for grouped-query attention
        # if there are fewer k/v heads than total heads, repeat them until the number matches
        if KVH < H:
            reps = H // KVH
            k = k[:,:,None,:,:].expand(B, KVH, reps, T, K).contiguous().view(B, H, T, K)
            v = v[:,:,None,:,:].expand(B, KVH, reps, T, V).contiguous().view(B, H, T, V)

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
    def __call__(self, x : Tensor, sublayer, layer_recurrent_memory : Optional[Tensor] = None):
        raise NotImplementedError

class ResidualAddOp(TransformerLayerPart, IResidualOp):
    def __init__(self, sublayer_norm_factory : Callable[..., nn.Module] = Factory(norm.RMSNorm, weight_scaling = False)):
        super().__init__()
        hparams = self.hparams
        self.dropout = nn.Dropout(hparams.dropout)
        self.norm = sublayer_norm_factory(hparams.d_model)

    def forward(self, x : Tensor, sublayer, layer_recurrent_memory : Optional[Tensor] = None):
        return x + self.dropout(sublayer(self.norm(x)))

class ResidualMixOp(TransformerLayerPart, IResidualOp):
    def __init__(self, sublayer_norm_factory : Callable[..., nn.Module] = Factory(norm.RMSNorm, weight_scaling = False)):
        super().__init__()
        hparams = self.hparams
        self.dropout = nn.Dropout(hparams.dropout)
        self.norm = sublayer_norm_factory(hparams.d_model)
        self.residual_mix = nn.Parameter(torch.ones(1,1,hparams.d_model))

    def forward(self, x : Tensor, sublayer, layer_recurrent_memory : Optional[Tensor] = None):
        return x * self.residual_mix + self.dropout(sublayer(self.norm(x))) * (2 - self.residual_mix)

class TransformerLayer(TransformerLayerPart):
    def __init__(self,  
                 self_attention_sublayer_factory : Callable[..., model.interface.IAttentionSubLayer] = Factory(AttentionSubLayer),
                 cross_attention_sublayer_factory : Callable[..., model.interface.IAttentionSubLayer | nn.Identity] = Factory(nn.Identity),
                 feedforward_sublayer_factory : Callable[..., model.interface.IFeedForwardSubLayer] = Factory(RWKVFeedForwardSubLayer),
                 residual_op_factory : Callable[..., IResidualOp] = Factory(ResidualMixOp, sublayer_norm_factory = Factory(norm.RMSNorm, weight_scaling = False)),
                 ):
        super().__init__()
        self.self_attention_sublayer = self_attention_sublayer_factory()
        self.self_attention_resop = residual_op_factory()

        self.cross_attention_sublayer = cross_attention_sublayer_factory()
        self.cross_attention_resop = residual_op_factory()

        self.feedforward_sublayer = feedforward_sublayer_factory()
        self.feedforward_resop = residual_op_factory()

    def forward(self, x : Tensor, encoder_output : Tensor | None = None, layer_recurrent_memory : Optional[Tensor] = None):
        # self attention (query, key, and value are all based on the same inputs)
        self_attn = lambda y: self.self_attention_sublayer(y, y, y, layer_recurrent_memory)
        x = self.self_attention_resop(x, self_attn) # this code looks a little complicated, but it's just allowing us to swap out whether we add or mix the residual

        # optional cross attention (query is based on the current input, but key and value are based on the encoder's output)
        # NOTE - we check against nn.Identity because the normal nn.Identity module does not allow extra arguments passed, though our interface.Identity version does
        if encoder_output is not None and not isinstance(self.cross_attention_sublayer, nn.Identity):
            cross_attn = lambda y: self.cross_attention_sublayer(y, encoder_output, encoder_output, layer_recurrent_memory)
            x = self.cross_attention_resop(x, cross_attn)

        # feedforward network
        feedforward = lambda y: self.feedforward_sublayer(y)
        x = self.feedforward_resop(x, feedforward)

        return x

class GradientCheckpointing(nn.Module):
    def __init__(self, module_factory : Callable[..., nn.Module]):
        super().__init__()
        self.module = module_factory()
    def forward(self, x, *_):
        x.requires_grad_(True)
        return torch.utils.checkpoint.checkpoint(self.module, x)

class Unembedding(nn.Module):
    def __init__(self, weight : Tensor = None):
        super().__init__()
        self.weight = weight

    def forward(self, x : Tensor):
        return F.linear(x, self.weight)

class Transformer(nn.Module):
    def __init__(self, 
                 hparams : HParams, 
                 embedding_norm_factory : Callable[..., nn.Module] = Factory(norm.RMSNorm, weight_scaling=False),
                 positional_embedding_factory : Callable[..., posemb.interface.IPositionalEmbedding | nn.Identity] = Factory(nn.Identity),
                 layer_factory : Callable[..., nn.Module] = Factory(TransformerLayer),
                 share_embedding_weights : bool = True,
                 final_norm_factory : Callable[..., nn.Module] = Factory(norm.RMSNorm, weight_scaling=False),
                 is_decoder : bool = True,
                 unembed_output : bool = True,
                 ):
        super().__init__()

        TransformerLayerPart.hparams = hparams

        self.embed = nn.Embedding(hparams.vocab_size, hparams.d_model)
        def smallInitEmbedding(m): nn.init.uniform_(self.embed.weight, a=-1e-4, b=1e-4)
        defer_module_init(self.embed, smallInitEmbedding) # SmallInit(Emb) per RWKV

        self.embedding_norm = embedding_norm_factory(hparams.d_model)
        self.positional_embedding = positional_embedding_factory(hparams.max_sequence_length, hparams.d_model)
        self.embedding_dropout = nn.Dropout(hparams.dropout)

        self.layers = nn.Sequential()
        for i in range(hparams.n_layer):
            TransformerLayerPart.layer_id = i
            self.layers.append(layer_factory())
        #self.layers = nn.Sequential(*[layer_factory() for i in range(hparams.n_layer)])

        if share_embedding_weights:
            self.final_norm = nn.Identity()
            self.unembed = Unembedding(self.embed.weight)
        else:
            self.final_norm = final_norm_factory(hparams.d_model)
            self.unembed = nn.Linear(hparams.d_model, hparams.vocab_size, bias = False)
        
        # FIXME - improve weight initialization somehow so it's not external to all the classes
        self._init_weights()

    def forward(self, x : Tensor, encoder_output : Tensor | None = None, recurrent_memory : Optional[list[Tensor]] = None):
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
        raise NotImplementedError()
    def decode(self, x : Tensor, encoder_output : Tensor | None = None, recurrent_memory : Optional[list[Tensor]] = None):
        raise NotImplementedError()

class Encoder(Transformer, IEncoderDecoder):
    def __init__(self, hparams : HParams, 
                 embedding_norm_factory : Callable[..., nn.Module] = Factory(norm.RMSNorm, weight_scaling=False),
                 positional_embedding_factory : Callable[..., posemb.interface.IPositionalEmbedding | nn.Identity] = Factory(nn.Identity),
                 layer_factory : Callable[..., nn.Module] = Factory(TransformerLayer),
                 share_embedding_weights : bool = True,
                 final_norm_factory : Callable[..., nn.Module] = Factory(norm.RMSNorm, weight_scaling=False),
):
        super().__init__(hparams, embedding_norm_factory, positional_embedding_factory, layer_factory, share_embedding_weights, final_norm_factory,  
                         is_decoder=False, unembed_output=True)

    def encode(self, x : Tensor):
        return self.forward(x)

class Decoder(Transformer, IEncoderDecoder):
    def __init__(self, hparams : HParams, 
                 embedding_norm_factory : Callable[..., nn.Module] = Factory(norm.RMSNorm, weight_scaling=False),
                 positional_embedding_factory : Callable[..., posemb.interface.IPositionalEmbedding | nn.Identity] = Factory(nn.Identity),
                 layer_factory : Callable[..., nn.Module] = Factory(TransformerLayer),
                 share_embedding_weights : bool = True,
                 final_norm_factory : Callable[..., nn.Module] = Factory(norm.RMSNorm, weight_scaling=False),
):
        super().__init__(hparams, embedding_norm_factory, positional_embedding_factory, layer_factory, share_embedding_weights, final_norm_factory, 
                         is_decoder=True, unembed_output=True)

    def decode(self, x : Tensor, encoder_output : Tensor | None = None, recurrent_memory : Optional[list[Tensor]] = None):
        return self.forward(x, encoder_output, recurrent_memory)

class EncoderDecoder(nn.Module, IEncoderDecoder):
    def __init__(self, hparams : HParams, 
                 embedding_norm_factory : Callable[..., nn.Module] = Factory(norm.RMSNorm, weight_scaling=False),
                 positional_embedding_factory : Callable[..., posemb.interface.IPositionalEmbedding | nn.Identity] = Factory(nn.Identity),
                 layer_factory : Callable[..., nn.Module] = Factory(TransformerLayer),
                 share_embedding_weights : bool = True,
                 final_norm_factory : Callable[..., nn.Module] = Factory(norm.RMSNorm, weight_scaling=False),
                 ):
        super().__init__()
        self.encoder = Transformer(hparams, embedding_norm_factory, positional_embedding_factory, layer_factory, share_embedding_weights, final_norm_factory, 
                                   is_decoder=False, unembed_output=False)
        self.decoder = Transformer(hparams, embedding_norm_factory, positional_embedding_factory, layer_factory, share_embedding_weights, final_norm_factory, 
                                   is_decoder=True, unembed_output=True)

    # used for training, not inference
    def forward(self, x : Tensor):
        x = self.encoder(x)
        x = self.decoder(torch.FloatTensor((0,0,0,0)), x)
        return x

    def encode(self, x : Tensor):
        return self.encoder(x)

    def decode(self, x : Tensor, encoder_output : Tensor | None = None, recurrent_memory : Optional[list[Tensor]] = None):
        return self.decoder(x, encoder_output, recurrent_memory)


