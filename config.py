from dataclasses import dataclass, field
import os
import torch


@dataclass
class BaseConfig:
    _dtype: torch.dtype = field(
        default=torch.float32, metadata={"help": "data type"}
    )  # WARN: Dtype Changed
    bsz: int = 256
    test_bsz: int = 1000

    epochs: int = 1000
    check_epoch: int = 1
    save_epoch: int = 10
    num_workers: int = os.cpu_count() - 2

    init_lr: float = 8e-3
    final_lr: float = 8e-4

    def __post_init__(self):
        self.device_setup()

    def device_setup(self):
        """Check Available Devices"""
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"


bargs = BaseConfig()
