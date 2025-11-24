import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import *
from config import bargs
from data import *


class ModelTrainer:
    """r
    Train and validate model
    """

    def __init__(self, model, train_loader, test_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.device = args.device

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.AdamW(
            model.parameters(), lr=args.init_lr, weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=args.final_lr
        )

    def run(self):
        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            for input, target in self.train_loader:
                input, target = input.to(self.device), target.to(self.device)

                output = self.model(input)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % self.args.check_epoch == 0:
                self.validate(epoch)

            if epoch % self.args.save_epoch == 0:
                torch.save(
                    self.model.state_dict(),
                    f"./path/" + bargs.model_name + f"_{epoch}.pth",
                )
                print("Model Successfully Saved!")

            self.scheduler.step()

    def validate(self, epoch):
        self.model.eval()
        correct_count = 0
        total_count = 0

        with torch.no_grad():
            for test_input, test_target in self.test_loader:
                test_input = test_input.to(self.device)
                test_target = test_target.to(self.device)
                test_output = self.model(test_input)

                preds = test_output.argmax(dim=1)
                correct_count += (preds == test_target).sum().item()
                total_count += test_target.size(0)

        accuracy = (correct_count / total_count) * 100
        print(f"Epoch {epoch}, Accuracy: {accuracy:.2f}%")

    def extract_layer(self, layer_num=27, w_bit=4, act_bit=4):
        print(f"Extracting data from Layer {layer_num}...")

        captured = {}

        def hook_curr_in(m, i):
            captured["in"] = i[0].detach()

        def hook_next_in(m, i):
            captured["next_in"] = i[0].detach()

        target_layer = self.model.features[layer_num]
        next_layer = self.model.features[layer_num + 2]

        h1 = target_layer.register_forward_pre_hook(hook_curr_in)
        h2 = next_layer.register_forward_pre_hook(hook_next_in)

        self.model.eval()
        images, _ = next(iter(self.test_loader))
        with torch.no_grad():
            self.model(images.to(self.device))

        h1.remove()
        h2.remove()

        w_alpha = target_layer.weight_quant.wgt_alpha
        w_scale = w_alpha / (2 ** (w_bit - 1) - 1)
        weight_int = torch.round(target_layer.weight_q / w_scale)

        act_alpha = target_layer.act_alpha
        a_scale = act_alpha / (2**act_bit - 1)
        act_quant_fn = act_quantization(act_bit)
        act_int = torch.round(act_quant_fn(captured["in"], act_alpha) / a_scale)

        output_int = F.conv2d(
            act_int[0].unsqueeze(0),
            weight_int,
            stride=target_layer.stride,
            padding=target_layer.padding,
        )
        output_int = F.relu(output_int)

        self._save_binary(weight_int, "./Files/weight_int.txt", w_bit)
        self._save_binary(act_int[0], "./Files/input_int.txt", act_bit)
        self._save_binary(
            output_int, "./Files/output_int.txt", 13 if act_bit == 2 else 15
        )

        output_recovered = output_int * w_scale * a_scale
        output_ref = captured["next_in"][0].unsqueeze(0)
        error = (output_recovered - output_ref).abs().mean()

        print(f"Files generated. Verification Error: {error.item():.6f}")

    def _save_binary(self, data, filename, bits):
        flat_data = data.detach().cpu().numpy().flatten().astype(int)
        mask = 2**bits - 1
        with open(filename, "w") as f:
            for num in flat_data:
                f.write(f"{num & mask:0{bits}b}\n")


if __name__ == "__main__":
    model = VGG_quant(
        vgg_name=bargs.model_name,
        weight_bits=bargs.weight_bits,
        act_bits=bargs.act_bits,
    ).to(bargs.device)
    model.load_state_dict(torch.load("./path/" + bargs.model_name + ".pth"))
    print(model)
    print(model.features[27])

    train_loader, test_loader = get_data_loaders()
    trainer = ModelTrainer(model, train_loader, test_loader, bargs)
    trainer.run()
    trainer.extract_layer(w_bit=bargs.weight_bits, act_bit=bargs.act_bits)
