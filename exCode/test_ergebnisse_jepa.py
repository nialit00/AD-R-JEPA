import torch

CKPT_PATH = "model_checkpoints/jepa_ssl_4_1e-4_18-25-48/model-50.pth"

ckpt = torch.load(CKPT_PATH, map_location="cpu")

print("\n====================")
print("TOP LEVEL KEYS")
print("====================")
print(ckpt.keys())

state = ckpt["model_state"]

print("\n====================")
print("MODEL STATE SAMPLE KEYS")
print("====================")

keys = list(state.keys())
print("Anzahl Parameter:", len(keys))

print("\nBeispiele:")
for k in keys[:30]:
    print(k)