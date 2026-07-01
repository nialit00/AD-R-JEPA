import torch

# JEPA laden
jepa_ckpt = torch.load(
    "model_checkpoints/jepa_ssl_4_1e-4_20-49-16/model-170000.pth",
    map_location="cpu"
)

# BEVCar laden
bev_ckpt = torch.load(
    "model_checkpoints/BEVCar_1x5_3e-4s_21-42-23/model-000071000.pth",
    map_location="cpu"
)

jepa_state = jepa_ckpt["model_state"]
bev_state = bev_ckpt["model_state_dict"]

print("\nVergleich Target Encoder → Radar Encoder\n")

for jepa_key in jepa_state:

    if not jepa_key.startswith("target_encoder.voxelnet."):
        continue

    bev_key = jepa_key.replace(
        "target_encoder.voxelnet.",
        "radar_encoder."
    )

    if bev_key in bev_state:

        if jepa_state[jepa_key].shape == bev_state[bev_key].shape:
            print("OK   ", bev_key, jepa_state[jepa_key].shape)
        else:
            print("SIZE MISMATCH")
            print(jepa_key, jepa_state[jepa_key].shape)
            print(bev_key, bev_state[bev_key].shape)

    else:
        print("MISSING:", bev_key)