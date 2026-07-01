import torch
import yaml
import sys
import os

sys.path.append(os.path.abspath("."))

from models.ad_r_jepa import AD_R_JEPA

# -----------------------------
# Load config
# -----------------------------
cfg = yaml.safe_load(open("configs/jepa/ad_r_jepa_nuscenes.yaml"))

model_cfg = cfg["MODEL"]
data_cfg = cfg["DATA_CONFIG"]

grid_size = data_cfg["POINT_CLOUD_RANGE_GRID_SIZE"]
voxel_size = data_cfg["VOXEL_SIZE"]
point_cloud_range = data_cfg["POINT_CLOUD_RANGE"]

# -----------------------------
# Build model
# -----------------------------
model = AD_R_JEPA(
    model_cfg=model_cfg,
    grid_size=grid_size,
    voxel_size=voxel_size,
    point_cloud_range=point_cloud_range
)

# -----------------------------
# Load checkpoint
# -----------------------------
ckpt = torch.load(
    "model_checkpoints/jepa_ssl_4_1e-4_19-50-17/model-000139000.pth",
    map_location="cpu"
)

missing, unexpected = model.load_state_dict(
    ckpt["model_state"],
    strict=False
)

# -----------------------------
# Report
# -----------------------------
print("\n====================")
print("LOAD CHECK")
print("====================")

print("\nMissing keys:", len(missing))
print("\nUnexpected keys:", len(unexpected))

print("\nExample missing:")
print(missing[:10])

print("\nExample unexpected:")
print(unexpected[:10])