import torch
from collections import OrderedDict

input_path = "./path/VGG16_bn_2bit.pth"
output_path = "./path/VGG16_bn_2bit_fixed.pth"

state_dict = torch.load(input_path, map_location="cpu")
new_state_dict = OrderedDict()

for key, value in state_dict.items():
    if ".weight_quant.wgt_alpha" in key:
        new_key = key.replace(".weight_quant.wgt_alpha", ".w_alpha")
        new_state_dict[new_key] = value

    elif key.endswith("weight_q"):
        continue

    else:
        new_state_dict[key] = value

torch.save(new_state_dict, output_path)
print(f"Fixed model saved to {output_path}")

# --- 验证环节 ---
print("\nVerifying keys in new model:")
found_alpha = False
for k in new_state_dict.keys():
    if "features.3.w_alpha" in k:
        print(f"✅ Found: {k}")
        found_alpha = True
        break
if not found_alpha:
    print("❌ Still missing features.3.w_alpha!")
