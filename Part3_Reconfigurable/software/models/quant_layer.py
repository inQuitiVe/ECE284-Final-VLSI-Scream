import torch.nn as nn
import torch
import torch.nn.functional as F


def signed_quantization(b, training=True):
    def uniform_quant(x, b):
        return x.mul(2**b - 1).round().div(2**b - 1)

    def int_quant(x, alpha):  # Only Used during inference
        x_c = x.div_(alpha).clamp(min=-1, max=1)
        sign = x_c.sign()
        return sign * x_c.abs().mul(2**b - 1).round()

    class _pass_through_quantization(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input.div_(alpha)
            input_c = input.clamp(min=-1, max=1)
            sign = input_c.sign()
            input_abs = input_c.abs()
            input_q = uniform_quant(input_abs, b).mul(sign)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)  # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()  # grad for weights will not be clipped
            input, input_q = ctx.saved_tensors
            i = (input.abs() > 1.0).float()
            sign = input.sign()
            grad_alpha = (grad_output * (sign * i + (input_q - input) * (1 - i))).sum()
            # grad_alpha = (grad_output * sign * i).sum() # Lower Accuracy
            grad_input = grad_input * (1 - i)
            return grad_input, grad_alpha

    return _pass_through_quantization().apply if training else int_quant


def unsigned_quantization(b, training=True):
    def uniform_quant(x, b):
        return x.mul(2**b - 1).round().div(2**b - 1)

    def int_quant(x, alpha):  # Only Used during inference
        return x.div_(alpha).clamp(max=1).mul(2**b - 1).round()

    class _unsigned_quantization(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input = input.div(alpha)
            input_c = input.clamp(max=1)
            input_q = uniform_quant(input_c, b)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            input, input_q = ctx.saved_tensors
            i = (input > 1.0).float()
            grad_alpha = (grad_output * (i + (input_q - input) * (1 - i))).sum()
            # grad_alpha = (grad_output * i).sum()
            grad_input = grad_input * (1 - i)
            return grad_input, grad_alpha

    return _unsigned_quantization().apply if training else int_quant


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit, training=True):
        super(weight_quantize_fn, self).__init__()
        self.w_bit = w_bit - 1
        self.weight_q = signed_quantization(self.w_bit, training)

    def forward(self, weight, w_alpha):
        mean = weight.data.mean()
        std = weight.data.std()
        weight = weight.add(-mean).div(std)  # weights normalization
        weight_q = self.weight_q(weight, w_alpha)

        return weight_q


class QuantConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        weight_bits=4,
        act_bits=4,
        unsigned=True,
    ):
        super(QuantConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.weight_quant = weight_quantize_fn(w_bit=weight_bits)
        if unsigned:
            self.act_quant = unsigned_quantization(act_bits)
        else:
            self.act_quant = signed_quantization(act_bits - 1)

        self.act_alpha = nn.Parameter(torch.tensor(8.0))
        self.w_alpha = nn.Parameter(torch.tensor(3.0))

    def forward(self, x):
        return F.conv2d(
            self.act_quant(x, self.act_alpha),
            self.weight_quant(self.weight, self.w_alpha),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QuantLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        weight_bits=4,
        act_bits=4,
        unsigned=True,
    ):
        super(QuantLinear, self).__init__(
            in_features,
            out_features,
            bias,
        )
        self.weight_quant = weight_quantize_fn(w_bit=weight_bits)

        if unsigned:
            self.act_quant = unsigned_quantization(act_bits)
        else:
            self.act_quant = signed_quantization(act_bits - 1)

        self.act_alpha = nn.Parameter(torch.tensor(8.0))
        self.w_alpha = nn.Parameter(torch.tensor(3.0))

    def forward(self, x):
        return F.linear(
            input=self.act_quant(x, self.act_alpha),
            weight=self.weight_quant(self.weight, self.w_alpha),
            bias=self.bias,
        )


class QuantConvNext(nn.Module):
    def __init__(
        self,
        dim,
        hidden_factor=4,
        layer_scale_init_value=1e-6,
        weight_bits=4,
        act_bits=4,
    ):
        super().__init__()

        self.dwconv = QuantConv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim,
            weight_bits=weight_bits,
            act_bits=act_bits,
            unsigned=True,
        )

        self.pwconv = nn.Sequential(
            Permute((0, 2, 3, 1)),
            nn.LayerNorm(dim, eps=layer_scale_init_value),
            QuantLinear(
                dim,
                hidden_factor * dim,
                weight_bits=weight_bits,
                act_bits=act_bits,
                unsigned=True,
            ),
            nn.ReLU(),
            QuantLinear(
                hidden_factor * dim,
                dim,
                weight_bits=weight_bits,
                act_bits=act_bits,
                unsigned=True,
            ),
        )
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x):
        return (self.gamma * (self.pwconv((self.dwconv(x))))).permute(0, 3, 1, 2) + x


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims).contiguous()
