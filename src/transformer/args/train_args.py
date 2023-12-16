from dataclasses import dataclass


@dataclass
class TTrainArgs:
    epochs: int
    lr: float
    lr_decay: float
    batch_size: int
    early_stoping: bool
    use_warmup: bool
