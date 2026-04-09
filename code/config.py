from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class EnvConfig:
    """Physical environment parameters for EdgeEnvV2."""

    n_nodes: int = 100
    topo_type: str = "ba"
    ba_m: int = 3
    dt_min: float = 1.0
    max_hours: float = 24.0
    lambda_base: float = 0.12
    alpha: float = 1.2
    demand_threshold_frac: float = 0.60
    capacity_range: Tuple[float, float] = (50.0, 500.0)
    init_util_range: Tuple[float, float] = (0.75, 0.90)
    throttle_min: float = 0.05
    throttle_cost_coeff: float = 0.02
    redundancy_factor: float = 1.0
    drop_fraction: float = 0.5

    @property
    def dt_hours(self) -> float:
        return self.dt_min / 60.0

    @property
    def max_steps(self) -> int:
        return int(round(self.max_hours / self.dt_hours))


@dataclass
class GATConfig:
    """Graph Attention encoder hyperparameters."""

    node_feat_dim: int = 4
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1


@dataclass
class SACConfig:
    """SAC and Lagrangian safety hyperparameters."""

    lr_actor: float = 1e-4
    lr_critic: float = 1e-4
    lr_alpha: float = 1e-4
    lr_lambda: float = 1e-4
    discount: float = 0.99
    tau: float = 0.005
    buffer_size: int = 500_000
    batch_size: int = 256
    init_alpha: float = 0.2
    target_entropy: float = -1.0
    hidden_dim: int = 256
    capacity_loss_limit: float = 0.30
    dropped_load_limit: float = 0.25


@dataclass
class TrainConfig:
    """Training loop configuration."""

    episodes: int = 500
    eval_scenarios: int = 40
    seed: int = 42
    device: str = "cpu"
    log_interval: int = 25
    save_interval: int = 100
    update_every: int = 10


@dataclass
class Config:
    """Top-level configuration container."""

    env: EnvConfig = field(default_factory=EnvConfig)
    gat: GATConfig = field(default_factory=GATConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
