import torch

# Checkpoint laden
ckpt = torch.load(
    "model_checkpoints/BEVCar_1x5_3e-4s_21-42-23/model-000071000.pth",
    map_location="cpu"
)

# Alle Layernamen holen
keys = list(ckpt["model_state_dict"].keys())

print("\n==============================")
print("Keys mit 'voxel'")
print("==============================")
for k in keys:
    if "voxel" in k.lower():
        print(k)

print("\n==============================")
print("Keys mit 'svfe'")
print("==============================")
for k in keys:
    if "svfe" in k.lower():
        print(k)

print("\n==============================")
print("Keys mit 'cml'")
print("==============================")
for k in keys:
    if "cml" in k.lower():
        print(k)

print("\n==============================")
print("Keys mit 'radar'")
print("==============================")
for k in keys:
    if "radar" in k.lower():
        print(k)