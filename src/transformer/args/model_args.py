from dataclasses import dataclass


@dataclass
class TModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    dropout_rate: float
    max_seq_length: int
