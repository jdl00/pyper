from dataclasses import dataclass


@dataclass
class MTrainArgs:
    # Number of experts
    n_experts: int

    # The limit of experts to select
    n_top_k: int
