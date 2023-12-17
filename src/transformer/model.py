import torch
import numpy as np

from math import sqrt
from torch import nn
from einops import reduce, rearrange
from typing import Optional, Tuple

from .args.model_args import TModelArgs

# Define new types for mask and cache
MaskType = Optional[torch.Tensor]
CacheType = Optional[Tuple[torch.Tensor, torch.Tensor]]


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        # Compute the root mean square value
        rms = torch.sqrt(reduce(x, "b l d -> b l", "mean", d=self.dim) + self.eps)

        # Normalize and scale
        return x / rms.unsqueeze(-1) * self.scale


class RoPE(nn.Module):
    def __init__(self, dims: int, traditional: bool = False, base: float = 10000):
        super().__init__()
        self.dims = dims
        self.traditional = traditional
        self.base = base

    def _extra_repr(self):
        return f"{self.dims}, traditional={self.traditional}"

    @staticmethod
    def create_cos_sin_theta(N, dims, offset=0, base=10000, dtype=torch.float32):
        position = torch.arange(offset, offset + N, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dims, 2).float() * (-np.log(base) / dims))
        theta = position * div_term
        return torch.cos(theta), torch.sin(theta)

    def _compute_rope(self, costheta, sintheta, x):
        x1, x2 = x[..., : self.dims // 2], x[..., self.dims // 2 : self.dims]
        rx1, rx2 = x1 * costheta - x2 * sintheta, x1 * sintheta + x2 * costheta

        if self.dims < x.shape[-1]:
            rx = torch.cat([rx1, rx2, x[..., self.dims :]], dim=-1)
        else:
            rx = torch.cat([rx1, rx2], dim=-1)

        return rx

    def _compute_traditional_rope(self, costheta, sintheta, x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        rx1, rx2 = x1 * costheta - x2 * sintheta, x1 * sintheta + x2 * costheta

        if self.dims < x.shape[-1]:
            raise NotImplementedError(
                "RoPE doesn't implement partial traditional application"
            )

        rx = rearrange([rx1, rx2], "two b n d -> b n (two d)")

        return rx

    def forward(self, x, offset: int = 0):
        shape = x.shape
        x = rearrange(x, "b n d -> (b n) d")
        N = x.shape[1] + offset
        costheta, sintheta = self.create_cos_sin_theta(
            N, self.dims, offset=offset, base=self.base, dtype=x.dtype
        )

        rope = (
            self._compute_traditional_rope if self.traditional else self._compute_rope
        )
        rx = rope(costheta, sintheta, x)

        return rx.reshape(shape)


class RoPEAttention(nn.Module):
    def __init__(self, args: TModelArgs):
        super().__init__()
        self.num_heads = args.n_heads
        self.rope = RoPE(args.dim // 2, traditional=False)
        self.Wqkv = nn.Linear(args.dim, 3 * args.dim)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.out_proj = nn.Linear(args.dim, args.dim)

    def forward(self, x: torch.Tensor, mask: MaskType = None, cache: CacheType = None):
        qkv = self.Wqkv(x)
        queries, keys, values = torch.chunk(qkv, 3, dim=-1)

        # Extract shapes and heads
        B, L, _ = x.shape
        num_heads = self.num_heads

        # Prepare the queries, keys, and values for the attention computation
        queries, keys, values = [
            rearrange(t, "b l (h d) -> b h l d", h=num_heads)
            for t in (queries, keys, values)
        ]

        # Add RoPE to the queries and keys and combine them with the cache
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = torch.cat([key_cache, keys], dim=2)
            values = torch.cat([value_cache, values], dim=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Perform the attention computation
        # TODO: Implement flash attention (allow for both mps and cuda)
        scale = sqrt(1 / queries.size(-1))
        scores = torch.matmul(queries * scale, keys.transpose(-2, -1))
        scores = self.dropout(scores)
        if mask is not None:
            scores += mask

        scores = torch.softmax(scores, dim=-1)
        values_hat = torch.matmul(scores, values)
        values_hat = rearrange(values_hat, "b h l d -> b l (h d)")

        return self.out_proj(values_hat), (keys, values)


class FeedForward(nn.Module):
    def __init__(self, args: TModelArgs):
        super().__init__()
        self.linear1 = nn.Linear(args.dim, args.hidden_dim)
        self.linear2 = nn.Linear(args.hidden_dim, args.dim)

        self.dropout = nn.Dropout(args.dropout_rate)
        self.norm = nn.LayerNorm(args.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the first linear transformation
        x = self.linear1(x)

        # Apply activation function (GELU in this case)
        x = nn.functional.gelu(x)

        # Apply dropout after activation
        x = self.dropout(x)

        # Apply layer normalization
        x = self.norm(x)

        # Apply the second linear transformation
        x = self.linear2(x)

        return x


class Block(nn.Module):
    def __init__(self, args: TModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = RoPEAttention(args=args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.args = args

    def forward(self, x: torch.Tensor, mask: MaskType = None, cache: CacheType = None):
        # Apply attention mechanism normalise x and pass to attention
        r, cache = self.attention(self.attention_norm(x), mask, cache)

        # Apply the dropout
        r = self.dropout(r)

        # Add the output of the attention mechanism to the input
        h = x + r

        # Apply the feed-forward network, normalise the attention output and
        # pass it to the feed forward network
        r = self.feed_forward(self.ffn_norm(h))

        # Add the output of the feed-forward network to the attention
        out = h + r

        # Return the output of the Transformer block and the updated cache.
        return out, cache


class Transformer(nn.Module):
    def __init__(self, args: TModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.dim = args.dim
        self.max_seq_length = args.max_seq_length

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.pos_embeddings = nn.Embedding(self.max_seq_length, args.dim)

        self.layers = nn.ModuleList([Block(args=args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor):
        # Assume 'x' is the input tensor with token indices, and 'positions'
        # is a tensor with position indices.

        # Add positional embeddings to token embeddings
        x = self.tok_embeddings(x) + self.pos_embeddings(positions)

        for layer in self.layers:
            x, _ = layer(x)

        x = self.norm(x)
        return self.output(x)


def summary(model: nn.Module):
    # TODO: Fefactor this to more closely match tf.summary()
    print(model)
