import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import *
from config import bargs
from data import *


def train(trainloader, testloader, model):
    criterion = nn.CrossEntropyLoss().to(bargs.device)
    optimizer = optim.AdamW(model.parameters(), lr=bargs.init_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=bargs.epochs, eta_min=bargs.final_lr
    )

    for epoch in range(1, bargs.epochs + 1):
        for input, target in trainloader:
            input, target = input.to(bargs.device), target.to(bargs.device)

            output = model(input)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % bargs.check_epoch == 0:
            validate(testloader, model, epoch)

        if epoch % bargs.save_epoch == 0:
            torch.save(model.state_dict(), "./path/VGG16_4bit_" + str(epoch) + ".pth")
            print(f"Model Successfully Saved!")

        scheduler.step()


def validate(testloader, model, epoch):
    print_total = epoch % bargs.save_epoch == 0
    mean_accuracy = []
    with torch.no_grad():
        for test_input, test_target in testloader:
            test_input, test_target = test_input.to(bargs.device), test_target.to(
                bargs.device
            )
            test_output = model(test_input)
            accuracy = (test_output.argmax(dim=1) == test_target).float().mean() * 100
            mean_accuracy.append(accuracy)
            if not print_total:
                print(f"Epoch {epoch}, Accuracy: {accuracy:.1f}%")
                return

        print(
            f"Epoch {epoch}, Total Accuracy: {torch.mean(torch.Tensor(mean_accuracy)):.1f}%"
        )


def check_psum(model, loader, device="cuda"):
    def psum_hook(module, input):
        x_float = input[0]
        weight_float = module.weight

        bit = module.bit
        w_bit = module.weight_quant.w_bit  # 实际上是 3 (4-1)
        w_alpha = module.weight_quant.wgt_alpha
        a_bit = bit  # 4
        a_alpha = module.act_alpha

        # 2. 计算整数权重 W_int
        mean = weight_float.data.mean()
        std = weight_float.data.std()
        w_norm = weight_float.add(-mean).div(std)
        w_div = w_norm.div(w_alpha)
        w_clamped = w_div.clamp(min=-1, max=1)
        w_scale_factor = 2**w_bit - 1
        w_int = (w_clamped * w_scale_factor).round()

        # 3. 计算整数输入 X_int
        x_div = x_float.div(a_alpha)
        x_clamped = x_div.clamp(max=1)  # ReLU 之后通常 min 已经是 0
        a_scale_factor = 2**a_bit - 1
        x_int = (x_clamped * a_scale_factor).round()

        # 4. 计算 Psum (整数卷积)
        psum_int = F.conv2d(
            x_int,
            w_int,
            bias=None,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

        # 5. 恢复 Psum (Recovered Psum)
        w_step = w_alpha / w_scale_factor
        a_step = a_alpha / a_scale_factor

        psum_recovered = psum_int * w_step * a_step

        # 6. 计算参考 Psum (Un-quantized / Float Psum)
        psum_ref = F.conv2d(
            x_float,
            w_norm,  # 使用 Normalize 后的 float 权重
            bias=None,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

        print("-" * 30)
        print(f"Layer: {module}")
        print(f"Checking Psum for Layer index 27...")
        print(f"Input shape: {x_int.shape}")
        print(f"X_int range: [{x_int.min().item()}, {x_int.max().item()}]")
        print(f"W_int range: [{w_int.min().item()}, {w_int.max().item()}]")

        mse = F.mse_loss(psum_recovered, psum_ref)
        print(f"MSE Loss between Recovered and Ref: {mse.item():.6f}")

        print(f"Recovered sample: {psum_recovered[0,0,0,:5].detach().cpu().numpy()}")
        print(f"Reference sample: {psum_ref[0,0,0,:5].detach().cpu().numpy()}")
        print("-" * 30)

    target_layer = model.features[27]
    hook_handle = target_layer.register_forward_hook(psum_hook)

    model.eval()
    with torch.no_grad():
        inputs, _ = next(iter(loader))
        inputs = inputs.to(device)
        model(inputs)

    hook_handle.remove()


if __name__ == "__main__":
    model = VGG_quant("VGG16_quant").to(bargs.device)
    model.load_state_dict(torch.load("./path/VGG16_4bit.pth"))
    # print(model)
    # print(model.features[27])
    trainloader, testloader = get_data_loaders()
    # check_psum(model, testloader)
    train(trainloader, testloader, model)
