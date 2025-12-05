import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


from models import *
from config import bargs
from data import *
from einops import rearrange


class ModelTrainer:
    """r
    Train and validate model;
    Extract a certain quantized layer into a specific format
    """

    def __init__(self, debug=True):
        model = ConvNext_quant(
            model_name=bargs.model_name,
            weight_bits=bargs.weight_bits,
            act_bits=bargs.act_bits,
        ).to(bargs.device)

        model_path = "./path/" + bargs.model_save_name + ".pth"
        if os.path.exists(model_path):
            model.load_state_dict(
                torch.load(
                    model_path,
                    map_location=bargs.device,
                )
            )

        train_loader, test_loader = get_data_loaders()

        if debug:
            print(model)
            print(
                f"{bargs.layer_num}th layer of model is "
                + str(model.features[bargs.layer_num])
            )

        self.best_accuracy = 0
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss().to(bargs.device)
        self.optimizer = optim.AdamW(
            model.parameters(), lr=bargs.init_lr, weight_decay=1e-5
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=bargs.epochs, eta_min=bargs.final_lr
        )

    def run(self):
        for epoch in range(1, bargs.epochs + 1):
            self.model.train()

            step = 0
            for input, target in self.train_loader:
                input, target = input.to(bargs.device), target.to(bargs.device)
                output = self.model(input)
                loss = self.criterion(output, target)
                loss.backward()
                step += 1

                if step % bargs.update_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            if step % bargs.update_steps != 0 and step > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if epoch % bargs.check_epoch == 0:
                self.validate(epoch)

            self.scheduler.step()

    def validate(self, epoch):
        self.model.eval()
        correct_count = 0
        total_count = 0

        with torch.no_grad():
            for test_input, test_target in self.test_loader:
                test_input = test_input.to(bargs.device)
                test_target = test_target.to(bargs.device)
                test_output = self.model(test_input)
                preds = test_output.argmax(dim=1)
                correct_count += (preds == test_target).sum().item()
                total_count += test_target.size(0)

        accuracy = (correct_count / total_count) * 100

        if accuracy > self.best_accuracy:
            torch.save(
                self.model.state_dict(),
                f"./path/" + bargs.model_save_name + f".pth",
            )
            self.best_accuracy = accuracy

        print(f"Epoch {epoch}, Accuracy: {accuracy:.2f}%")

    def extract_layer(self, layer_num=27, w_bit=4, act_bit=4):
        print(f"Extracting data from Layer {layer_num}...")

        # Only take 1st image of the batch
        captured = {}

        def hook_curr_in(m, i):
            captured["in"] = i[0].detach()

        def hook_output(m, i):
            captured["output"] = i[0].detach()

        target_layer = self.model.features[layer_num]
        next_layer = self.model.features[layer_num + 2]
        h1 = target_layer.register_forward_pre_hook(hook_curr_in)
        h2 = next_layer.register_forward_pre_hook(hook_output)
        self.model.eval()

        images, _ = next(iter(self.test_loader))
        with torch.no_grad():
            self.model(images.to(bargs.device))
        h1.remove()
        h2.remove()

        # 1. Quantization
        w_alpha = target_layer.weight_quant.wgt_alpha
        w_scale = w_alpha / (2 ** (w_bit - 1) - 1)
        weight_int = torch.round(target_layer.weight_q / w_scale)
        input_float = captured["in"][0]

        act_alpha = target_layer.act_alpha
        a_scale = act_alpha / (2**act_bit - 1)
        act_quant_fn = act_quantization(act_bit)
        act_int = torch.round(act_quant_fn(input_float, act_alpha) / a_scale)

        # 2. Conv
        output_int = F.relu_(
            F.conv2d(
                act_int.unsqueeze(0),
                weight_int,
                stride=target_layer.stride,
                padding=target_layer.padding,
            )
        ).squeeze(0)

        # 3. Save Files

        if bargs.pe_config == "ws":
            for i in range(bargs.tile_image_size):
                self._save_file(
                    data=rearrange(
                        F.pad(act_int.unsqueeze(0), pad=(1, 1, 1, 1), value=0),
                        "1 (th ts) h w -> th ts (h w)",
                        th=bargs.tile_image_size,
                        ts=bargs.tile_size,
                    )[i],
                    filename="./Files/"
                    + str(act_bit)
                    + "bit/"
                    + bargs.pe_config
                    + "/activation_tile"
                    + str(i)
                    + ".txt",
                    bits=act_bit,
                )

            for i in range(bargs.tile_image_size**2):
                for j in range(9):
                    self._save_file(
                        data=rearrange(
                            weight_int,
                            "(th tsh) (tw tsw) h w -> (tw th) tsh tsw (h w)",
                            tsh=bargs.tile_size,
                            tsw=bargs.tile_size,
                        )[i, :, :, j],
                        filename="./Files/"
                        + str(act_bit)
                        + "bit/"
                        + bargs.pe_config
                        + "/weight_tile_"
                        + str(i)
                        + "_kij_"
                        + str(j)
                        + ".txt",
                        bits=w_bit,
                    )

        elif bargs.pe_config == "os":
            act_padded = F.pad(act_int.unsqueeze(0), (1, 1, 1, 1))
            for i in range(bargs.channel // bargs.tile_size):
                for j in range(2):
                    self._save_file(
                        data=act_padded[
                            0,
                            bargs.tile_size * i : bargs.tile_size * i + bargs.tile_size,
                            2 * j : 2 * j + 4,
                            :,
                        ].reshape(bargs.tile_size, -1),
                        filename=(
                            "./Files/"
                            + str(act_bit)
                            + "bit/"
                            + bargs.pe_config
                            + "/channel_group_"
                            + str(i)
                            + ("_upper" if j == 0 else "_lower")
                            + ".txt"
                        ),
                        bits=act_bit,
                    )

                reshaped_weight = rearrange(
                    weight_int,
                    "(tn par_cout) (ts par_cin) k1 k2 -> tn par_cin (k1 k2) ts par_cout",
                    tn=bargs.tile_image_size,
                    ts=bargs.tile_size,
                )
                for tn in range(bargs.tile_image_size):
                    for kij in range(9):
                        self._save_file(
                            data=reshaped_weight[tn, i, kij, :, :].reshape(8, -1),
                            filename="./Files/"
                            + str(act_bit)
                            + "bit/"
                            + bargs.pe_config
                            + "/weight_channel_group_"
                            + str(i)
                            + "_kij_"
                            + str(kij)
                            + "_tile_"
                            + str(tn)
                            + ".txt",
                            bits=w_bit,
                        )

        for i in range(bargs.tile_image_size):
            self._save_file(
                data=rearrange(
                    output_int, "(tn ts) h w -> tn ts (h w)", ts=bargs.tile_size
                )[i],
                filename="./Files/" + str(act_bit) + "bit/output_" + str(i) + ".txt",
                bits=16,
            )

        # 4. Error Calculation
        output_recovered = output_int * w_scale * a_scale
        output_ref = captured["output"][0]
        error = (output_recovered - output_ref).abs().mean()

        print(f"Files generated. Verification Error: {error.item():.6f}")

    def _save_file(self, data, filename, bits):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        file = open(filename, "w")
        file.write("#time0row7[msb-lsb],time0row6[msb-lst],....,time0row0[msb-lst]#\n")
        file.write("#time1row7[msb-lsb],time1row6[msb-lst],....,time1row0[msb-lst]#\n")
        file.write("#................#\n")
        fmt_str = f"0{bits}b"

        for i in range(data.size(1)):
            for j in range(data.size(0)):
                data_value = round(data[7 - j, i].item())
                if data_value < 0:
                    data_bin = format(data_value & (2**bits - 1), fmt_str)
                else:
                    data_bin = format(data_value, fmt_str)
                for k in range(bits):
                    file.write(data_bin[k])
            file.write("\n")
        file.close()


if __name__ == "__main__":
    trainer = ModelTrainer(debug=True)
    trainer.run()
    # trainer.extract_layer(
    #     layer_num=bargs.layer_num, w_bit=bargs.weight_bits, act_bit=bargs.act_bits
    # )
