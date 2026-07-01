


import torch


jepa_ckpt_path = "model_checkpoints/jepa_ssl_4_1e-4_19-50-17/model-000154000.pth"
bevcar_ckpt_path = "model_checkpoints/BEVCarFinetune_1x5_3e-4s_17-25-15/model-000001000.pth"

'''
--------------------------------------------------
Weights: Compare JEPA + BEVCar to JEPA
---------------------------------------------------
jepa_raw = torch.load(jepa_ckpt_path, map_location="cpu")
jepa_state = jepa_raw.get("model_state_dict", jepa_raw.get("model_state"))

bevcar_raw = torch.load(bevcar_ckpt_path, map_location="cpu")
bevcar_state = bevcar_raw["model_state_dict"]

radar_keys = [k for k in bevcar_state.keys() if "radar" in k or "voxel" in k]

print("\n=== Vergleich Radar-Encoder (nur echte Gewichte) ===\n")

for k in radar_keys[:20]:
    if k not in jepa_state:
        continue

    j = jepa_state[k]
    b = bevcar_state[k]

    # Nur Float-Tensoren vergleichen
    if not torch.is_floating_point(j):
        continue

    diff = (b - j).abs().mean().item()

    print(f"{k}")
    print("  Diff:", diff)
'''

'''
-----------------------------------
Weights: Compare JEPA + BEVCar to BEVCar
-----------------------------------
'''

scratch_ckpt = "model_checkpoints/BEVCar_1x5_3e-4s_19-21-10/model-000011000.pth"
pretrain_ckpt = "model_checkpoints/BEVCarFinetune_1x5_3e-4s_17-25-15/model-000011000.pth"

# -------------------------------------------------------
# Laden
# -------------------------------------------------------
scratch_raw = torch.load(scratch_ckpt, map_location="cpu")
pretrain_raw = torch.load(pretrain_ckpt, map_location="cpu")

scratch_state = scratch_raw["model_state_dict"]
pretrain_state = pretrain_raw["model_state_dict"]

# -------------------------------------------------------
# Radar-Parameter finden
# -------------------------------------------------------
radar_keys = [k for k in scratch_state.keys() if "radar" in k or "voxel" in k]

print("\n=== Vergleich: BEVCar Scratch vs. BEVCar mit JEPA ===\n")

for k in radar_keys[:20]:
    s = scratch_state[k]
    p = pretrain_state[k]

    # Nur Float-Tensoren vergleichen
    if not torch.is_floating_point(s):
        continue

    diff = (p - s).abs().mean().item()

    print(f"{k}")
    print("  Diff:", diff)
