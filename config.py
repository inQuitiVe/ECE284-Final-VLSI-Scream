from dataclasses import dataclass, field
import os
import torch


@dataclass
class BaseConfig:
    train_bsz: int = 128
    test_bsz: int = 500

    epochs: int = 100
    check_epoch: int = 5
    num_workers: int = os.cpu_count() - 2

    init_lr: float = 5e-3
    final_lr: float = 4e-4

    tile_size: int = 8
    pe_config: str = field(default="ws", metadata={"choices": ["os", "ws"]})
    model_config: str = field(
        default="bn",
        metadata={"choices": ["bn", "standard"]},
    )
    model_name: str = field(
        default="VGG16", metadata={"choices": ["VGG16", "ConvNext"]}
    )

    update_steps: int = 1
    # cosine_point: float = 0.7
    # flat_point: float = 0.2

    weight_bits: int = 4
    act_bits: int = 2

    def __post_init__(self):
        self.setup()

    def setup(self):
        # Select Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Select Channel
        if self.act_bits == 2:
            self.channel = 16
        elif self.act_bits == 4:
            self.channel = 8

        # Select layer_num
        if self.model_name == "VGG16":
            self.layer_num = 27
        elif self.model_name == "ConvNext":
            self.layer_num = 11

        # Select Model Names
        self.model_save_name = (
            self.model_name + "_" + self.model_config + "_" + str(self.act_bits) + "bit"
        )
        self.model_name = self.model_name + "_" + str(self.act_bits) + "bit"
        self.tile_image_size = self.channel // self.tile_size


bargs = BaseConfig()
