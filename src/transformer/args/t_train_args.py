from dataclasses import dataclass


@dataclass
class TTrainArgs:
    epochs: int
    lr: float
    lr_decay: float
    batch_size: int
    early_stopping: bool
    use_warmup: bool
    gradient_accumulation_steps: int
    step_size: int
    gamma: float
