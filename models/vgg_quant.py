import torch
import torch.nn as nn
from .quant_layer import *

cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16_4bit": [
        "F",
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        8,
        8,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG16_2bit": [
        "F",
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        16,
        16,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG16": [
        "F",
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG_quant(nn.Module):
    def __init__(self, vgg_name, weight_bits=4, act_bits=4):
        super(VGG_quant, self).__init__()
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.no_bn = False

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == "F":
                layers += [
                    nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ]
                in_channels = 64
            elif x == 8 or x == 16:
                if self.no_bn:
                    layers += [
                        QuantConv2d(
                            in_channels,
                            x,
                            kernel_size=3,
                            padding=1,
                            weight_bits=self.weight_bits,
                            act_bits=self.act_bits,
                        ),
                        nn.ReLU(inplace=True),
                    ]
                    self.no_bn = False
                else:
                    layers += [
                        QuantConv2d(
                            in_channels,
                            x,
                            kernel_size=3,
                            padding=1,
                            weight_bits=self.weight_bits,
                            act_bits=self.act_bits,
                        ),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True),
                    ]
                    self.no_bn = True

                in_channels = x
            else:
                layers += [
                    QuantConv2d(
                        in_channels,
                        x,
                        kernel_size=3,
                        padding=1,
                        weight_bits=self.weight_bits,
                        act_bits=self.act_bits,
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def fuse_model(self):
        new_features = []
        i = 0
        while i < len(self.features):
            layer = self.features[i]
            if (
                i + 1 < len(self.features)
                and isinstance(layer, (nn.Conv2d, QuantConv2d))
                and isinstance(self.features[i + 1], nn.BatchNorm2d)
            ):

                conv = layer
                bn = self.features[i + 1]

                with torch.no_grad():
                    mu = bn.running_mean
                    sigma = torch.sqrt(bn.running_var + bn.eps)
                    gamma = bn.weight
                    beta = bn.bias

                    w = conv.weight
                    if conv.bias is not None:
                        b = conv.bias
                    else:
                        b = torch.zeros_like(mu)

                    w_new = w * (gamma / sigma).reshape(-1, 1, 1, 1)
                    b_new = beta + (b - mu) * (gamma / sigma)

                    conv.weight.data.copy_(w_new)
                    if conv.bias is None:
                        conv.bias = nn.Parameter(b_new)
                    else:
                        conv.bias.data.copy_(b_new)

                new_features.append(conv)
                i += 2
            else:
                new_features.append(layer)
                i += 1

        return nn.Sequential(*new_features)
