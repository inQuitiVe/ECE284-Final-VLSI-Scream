from dataclasses import dataclass, field
import os
import torch


@dataclass
class BaseConfig:
    train_bsz: int = 256
    test_bsz: int = 1000

    epochs: int = 140
    check_epoch: int = 5
    save_epoch: int = 5
    num_workers: int = os.cpu_count() - 2

    init_lr: float = 5e-3
    final_lr: float = 4e-4

    tile_size: int = 8
    pe_config: str = "ws"  # os

    # update_steps: int = 5
    # cosine_point: float = 0.7
    # flat_point: float = 0.2

    weight_bits: int = 4
    act_bits: int = 2

    def __post_init__(self):
        self.setup()

    def setup(self):
        """Check Available Devices, Select Model Name"""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if self.act_bits == 2:
            self.model_name = "VGG16_2bit"
            self.channel = 16
        elif self.act_bits == 4:
            self.model_name = "VGG16_4bit"
            self.channel = 8

        self.tile_image_size = self.channel // self.tile_size


bargs = BaseConfig()
